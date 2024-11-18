import argparse
from pathlib import Path
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "export_dir",
        type=Path,
        help=(
            "Path to the directory containing the .txt versions of the sparse "
            "COLMAP reconstruction."
        )
    )
    parser.add_argument(
        "--q",
        type=float,
        required=False,
        help=(
            "Quantile used as an upper bound for inliers. All points that are "
            "further away than the closest q * num points will be not be "
            "considered in the determination of the bounding box."
        ),
        default=0.001
    )
    args = parser.parse_args()

    q = args.q
    points_path = args.export_dir / "points3D.txt"
    images_path = args.export_dir / "images.txt"

    # read 3D points

    point_ids = []
    points = []

    with points_path.open() as file:
        # skip first 3 lines
        [file.readline() for _ in range(3)]

        for line in file.readlines():
            pid, x, y, z, _ = line.strip().split(" ", maxsplit=4)
            v = np.array([float(x), float(y), float(z)])
            points.append(v)
            point_ids.append(int(pid))

    points = np.stack(points, axis=0)
    point_ids = np.array(point_ids)
    point_id_map = {k: i for i, k in enumerate(point_ids)}
    
    # read image file

    min_cam_point_dist = np.inf
    max_cam_point_dist = -np.inf

    with images_path.open() as file:
        # skip first 3 lines
        [file.readline() for _ in range(4)]

        while (line := file.readline()) != "":
            #   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME
            #   POINTS2D[] as (X, Y, POINT3D_ID)
            tx, ty, tz = line.strip().split(' ')[5:8]
            cam_xyz = np.array([float(tx), float(ty), float(tz)])

            flat_point_tuples = file.readline().strip().split()
            assoc_point_ids = [int(idx) for i, idx in enumerate(flat_point_tuples) if i % 3 == 2 and idx != "-1"]
            mapped_point_ids = np.array([point_id_map[i] for i in assoc_point_ids])
            assoc_points = points[mapped_point_ids]
            dists_to_cam = np.linalg.norm(assoc_points - cam_xyz[None], axis=1)
            min_quant_dist, max_quant_dist = np.quantile(dists_to_cam, q=(q, 1 - q), axis=0)
            min_cam_point_dist = min(min_cam_point_dist, min_quant_dist)
            max_cam_point_dist = max(max_cam_point_dist, max_quant_dist)

    print(f"{q:6.1%} distance:", min_cam_point_dist)
    print(f"{1-q:6.1%} distance:", max_cam_point_dist)

    scale = 0.15 / min_cam_point_dist
    quant_bounds = np.quantile(points, q=(q, 1 - q), axis=0)
    quant_center = quant_bounds.mean(axis=0)

    print()
    print("Putting closest point 0.15 units away, yields")
    print("near   = 0.1 (fixed)")
    print("far    =", max_cam_point_dist * scale)
    print("scale  =", scale)
    print("center =", np.array2string(quant_center, separator=','))
    print()
    print(f"{1 - q:6.1%} of points will be in this bounding box:")
    half_lengths = (quant_bounds[1] - quant_center) * scale
    print(' x '.join(f"[{-h:.3f}, {h:.3f}]" for h in half_lengths))
    print()
    

if __name__ == "__main__":
    main()