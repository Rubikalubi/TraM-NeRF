import argparse
from pathlib import Path
import warnings
import numpy as np
import json
import shutil
import imageio.v2 as imageio
from skimage.transform import downscale_local_mean
from tqdm import tqdm

def qvec2rotmat(qvec):
    # from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/utils/colmap_parsing_utils.py#L454
    return np.array(
        [
            [
                1 - 2 * qvec[2] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[1] * qvec[2] - 2 * qvec[0] * qvec[3],
                2 * qvec[3] * qvec[1] + 2 * qvec[0] * qvec[2],
            ],
            [
                2 * qvec[1] * qvec[2] + 2 * qvec[0] * qvec[3],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[3] ** 2,
                2 * qvec[2] * qvec[3] - 2 * qvec[0] * qvec[1],
            ],
            [
                2 * qvec[3] * qvec[1] - 2 * qvec[0] * qvec[2],
                2 * qvec[2] * qvec[3] + 2 * qvec[0] * qvec[1],
                1 - 2 * qvec[1] ** 2 - 2 * qvec[2] ** 2,
            ],
        ]
    )

def convert_colmap_to_nerf_dataset(
        source_dir: Path,
        output_dir: Path,
        images_dir: Path,
        masks_dir: Path,
        test_ratio: float,
        downscale_factor: int,
        downscale_images: bool
    ):

    # parse camera intrinsics
    print("\n", "reading camera intrinsics...")
    
    with (source_dir / "cameras.txt").open() as camera_file:
        # skip first 3 lines
        [camera_file.readline() for _ in range(3)]

        # the relevant line looks like this:
        # 1 SIMPLE_RADIAL 4608 3456 3852.2552728649166 2304 1728 -0.025608934207212677

        camera_info = camera_file.readline().strip().split(" ")
        camera_id = int(camera_info[0])
        distortion_model = camera_info[1]
        width, height = int(camera_info[2]), int(camera_info[3])

        if distortion_model == 'SIMPLE_RADIAL':
            fx = fy = float(camera_info[4]) / downscale_factor
            cx = float(camera_info[5]) / downscale_factor
            cy = float(camera_info[6]) / downscale_factor
        elif distortion_model == 'PINHOLE':
            fx = float(camera_info[4]) / downscale_factor
            fy = float(camera_info[5]) / downscale_factor
            cx = float(camera_info[6]) / downscale_factor
            cy = float(camera_info[7]) / downscale_factor
        else:
            raise NotImplementedError(f"Parser for distortion model '{distortion_model}' not implemented.")

        # check if data is as expected
        assert camera_id == 1


        camera_data = {
            "camera_angle_x": 2 * np.arctan(width / 2 / fx),
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            # "k1": k1,
            "intrinsics": [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0]
            ]
        }

    print(camera_data)

    # parse camera extrinsics
    # Image list with two lines of data per image:
    #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
    #   POINTS2D[] as (X, Y, POINT3D_ID)
    # Number of images: 264, mean observations per image: 708.70833333333337
    print("\n", "reading camera extrinsics...")

    frame_data = []

    with (source_dir / "images.txt").open() as images_file:
        # skip first 4 lines
        [images_file.readline() for _ in range(4)]

        # read every other line
        lines = list(images_file.readlines())
        for line in lines[::2]:
            image_info = line.strip().split(" ")
            image_id = int(image_info[0])
            q = np.array([float(v) for v in image_info[1:5]])
            t = np.array([float(v) for v in image_info[5:8]])
            image_camera_id = int(image_info[8])
            image_name = image_info[9]

            assert image_camera_id == 1
            
            # adapted from https://github.com/nerfstudio-project/nerfstudio/blob/main/nerfstudio/data/dataparsers/colmap_dataparser.py#L123
            rotation = qvec2rotmat(q)
            translation = t.reshape(3, 1)
            w2c = np.concatenate([rotation, translation], 1)
            w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
            c2w = np.linalg.inv(w2c)
            # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
            c2w[0:3, 1:3] *= -1
            # Why do we want to flip Z with a handedness transform?
            # See https://github.com/nerfstudio-project/nerfstudio/issues/1504
            c2w = c2w[np.array([1, 0, 2, 3]), :]
            c2w[2, :] *= -1

            frame_data.append({
                "file_path": image_name,
                "transform_matrix": [[float(v) for v in r] for r in c2w]
            })

    print("Found", len(frame_data), "valid images with extrinsics")

    # generate indices for train / test split
    assert 0.0 <= test_ratio <= 1.0
    num_images = len(frame_data)
    num_test_images = int(test_ratio * num_images)
    test_indices = np.random.permutation(num_images)[:num_test_images]

    # create output dirs
    (output_dir / "train").mkdir(parents=True, exist_ok=False)
    (output_dir / "test").mkdir(parents=False, exist_ok=False)

    # split output_data into train and test
    output_train_data = {**camera_data, "frames": []}
    output_test_data = {**camera_data, "frames": []}

    expected_shape = (height // downscale_factor, width // downscale_factor, 3)
    print("Images should have shape", expected_shape)

    if masks_dir is not None:
        output_masks_dir = output_dir / "masks"
        output_masks_dir.mkdir()

        print("\n", "Copying masks to output directory...")

        for path in tqdm(list(masks_dir.glob("*.png"))):
            mask = imageio.imread(path)

            if downscale_images:
                mask = downscale_local_mean(mask, (downscale_factor, downscale_factor, 1)).astype(np.uint8)
            
            imageio.imwrite(output_masks_dir / path.name, mask)

    print("\n", "Copying images to output directory...")

    for i in tqdm(range(num_images), disable=not downscale_images):
        d = frame_data[i]
        image_name = d["file_path"]
        if i in test_indices:
            d["file_path"] = "test/" + image_name
            output_test_data["frames"].append(d)
        else:
            d["file_path"] = "train/" + image_name
            output_train_data["frames"].append(d)
        
        # copy into target image dir
        source_image_path = images_dir / image_name

        image = imageio.imread(source_image_path, rotate=True)

        if downscale_images:
            image = downscale_local_mean(image, factors=(downscale_factor, downscale_factor, 1)).astype(np.uint8)

        imageio.imwrite(output_dir / d['file_path'], image)

        if image.shape != expected_shape:
            # check for the nerfstudio off-by-one downsampling bug
            if abs(image.shape[0] - height // downscale_factor) < 2 or abs(image.shape[1] - width // downscale_factor) < 2:
                warnings.warn(RuntimeWarning(
                    f"Downscaled image size is off by one pixel. Expected {expected_shape}, but "
                    f"got {image.shape}. This is known to be caused by the downsampling in "
                    f"nerfstudio."
                ))
            else:
                raise Exception(
                    f"Image shapes are inconsistent. Expected {expected_shape}, "
                    f"but got {image.shape} for '{source_image_path}'.")
            
    # write json files
    with (output_dir / "transforms_train.json").open("x") as train_file:
        json.dump(output_train_data, train_file, indent=4)
    with (output_dir / "transforms_test.json").open("x") as test_file:
        json.dump(output_test_data, test_file, indent=4)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source_dir",
        type=Path,
        help="COLMAP export dir containing .txt versions of the "
             "reconstructions."
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        help="Directory to save the Blender formatted scene into."
    )
    parser.add_argument(
        "--images_dir",
        type=Path,
        help="Path to the (undistorted) images of the dataset."
    )
    parser.add_argument(
        "--masks_dir",
        type=Path,
        default=None,
        help="(optional) Path to masks used in COLMAP."
    )
    parser.add_argument(
        "--test_ratio",
        type=float,
        help="Ratio with which to split images into train and test set.",
        default=0.2
    )
    parser.add_argument(
        "--downscale_factor",
        type=int,
        default=1,
        help="Factor to downscale the intrinsics with."
    )
    parser.add_argument(
        "--downscale_images",
        action='store_true',
        help="If set, will also downscale the images (and masks)."
    )
    args = parser.parse_args()
    convert_colmap_to_nerf_dataset(**args.__dict__)

if __name__ == "__main__":
    main()
    