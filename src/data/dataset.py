# ------------------------------------------------------------------------------------
# Modified from NerfAcc (https://github.com/nerfstudio-project/nerfacc)
# Copyright (c) 2022 Ruilong Li, UC Berkeley.
# ------------------------------------------------------------------------------------

import json
import os
import math
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import pyexr

from src.data.utils import Rays

radii_factor = 2 / math.sqrt(12)

def ndc_rays(H, W, focal, near, rays_o, rays_d):
    """Normalized device coordinate rays.

    Space such that the canvas is a cube with sides [-1, 1] in each axis.

    Args:
      H: int. Height in pixels.
      W: int. Width in pixels.
      focal: float. Focal length of pinhole camera.
      near: float or array of shape[batch_size]. Near depth bound for the scene.
      rays_o: array of shape [batch_size, 3]. Camera origin.
      rays_d: array of shape [batch_size, 3]. Ray direction.

    Returns:
      rays_o: array of shape [batch_size, 3]. Camera origin in NDC.
      rays_d: array of shape [batch_size, 3]. Ray direction in NDC.
    """
    # Shift ray origins to near plane
    t = -(near + rays_o[..., 2]) / rays_d[..., 2]
    rays_o = rays_o + t[..., None] * rays_d

    # Projection
    o0 = -1./(W/(2.*focal)) * rays_o[..., 0] / rays_o[..., 2]
    o1 = -1./(H/(2.*focal)) * rays_o[..., 1] / rays_o[..., 2]
    o2 = 1. + 2. * near / rays_o[..., 2]

    d0 = -1./(W/(2.*focal)) * \
        (rays_d[..., 0]/rays_d[..., 2] - rays_o[..., 0]/rays_o[..., 2])
    d1 = -1./(H/(2.*focal)) * \
        (rays_d[..., 1]/rays_d[..., 2] - rays_o[..., 1]/rays_o[..., 2])
    d2 = -2. * near / rays_o[..., 2]

    rays_o = torch.stack([o0, o1, o2], -1)
    rays_d = torch.stack([d0, d1, d2], -1)

    return rays_o, rays_d


def read_image(file_path: Path | str):
    #file_path = Path(file_path)
    file_path = Path(str(file_path).replace("\\", "/"))
    
    try:
        # try to read as png
        if file_path.suffix == ".exr":
            raise Exception
        if file_path.suffix == "":
            file_path = file_path.with_suffix(".png")
        rgba = imageio.imread(str(file_path))
        depth = normal = None
    except Exception:
        # try to read as exr
        file_path = file_path.with_suffix(".exr")

        with pyexr.open(str(file_path)) as file:
            rgba = file.get("ViewLayer.Combined")
            depth = file.get("ViewLayer.Depth")
            normal = file.get("ViewLayer.Normal")

        # map rgba from linear to sRGB
        rgba = np.clip(rgba, 1e-10, 1.0)
        rgba = np.where(
            rgba <= 0.0031308,
            12.92 * rgba,
            1.055 * rgba ** (1.0 / 2.4) - 0.055
        ) * 255
    
    return rgba, depth, normal


def _load_renderings(root_fp: str, subject_id: str, split: str):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "..", "..", root_fp,
        )

    data_dir = os.path.join(root_fp, subject_id)
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    depths = []
    normals = []
    camtoworlds = []
    paths = []

    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        if "." not in Path(frame["file_path"]).name:
            fname = os.path.join(data_dir, frame["file_path"] + ".png")
        else:
            fname = os.path.join(data_dir, frame["file_path"])
        paths.append(fname)
        rgba, depth, normal = read_image(fname)
        if rgba.shape[2] == 3:
            a = np.ones_like(rgba[..., :1]) * 255
            rgba = np.concatenate([rgba, a], axis=-1)
        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)

        if depth is not None:
            depths.append(depth)
        if normal is not None:
            normals.append(normal)

    images = np.stack(images, axis=0)

    depths = np.stack(depths, axis=0) if len(depths) > 0 else None
    normals = np.stack(normals, axis=0) if len(normals) > 0 else None
    camtoworlds = np.stack(camtoworlds, axis=0)
    assert images.shape[-1] == 4
    h, w = images.shape[1:3]

    if "intrinsics" in meta:
        K = np.array(meta["intrinsics"])
    else:
        camera_angle_x = float(meta["camera_angle_x"])
        cx = float(meta.get("cx", w / 2))
        cy = float(meta.get("cy", h / 2))
        fx = fy = 0.5 * w / np.tan(0.5 * camera_angle_x)
        K = np.array([
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0]
        ])
    
    return images, paths, camtoworlds, K, depths, normals

class SubjectLoader(torch.utils.data.Dataset):
    """Single subject data loader for training and evaluation."""
    SPLITS = ["train", "val", "trainval", "test"]
    SUBJECT_IDS = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
        "results_300",
        "different_camera_angles",
        "faker",
        "faker-better",
        "glossy",
        "small_mirror",
        "rotated",
        "small",
        "small_reloaded",
        "small_complex"
    ]
    # WIDTH, HEIGHT = 800, 800
    # NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = True

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        get_radii: bool = True,
        ndc = False,
        scale: float = 1.0,
        center: torch.Tensor = None,
        patch_size: int = None
    ):
        super().__init__()
        assert split in self.SPLITS, "%s" % split
        #assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        self.ndc = ndc
        self.split = split
        self.num_rays = num_rays
        self.near = near
        self.far = far
        self.training = (num_rays is not None) and (
            split in ["train", "trainval"]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.get_radii = get_radii
        self.image_paths = []
        self.scale = scale
        self.patch_size = patch_size

        if self.training:
            assert num_rays % (patch_size ** 2) == 0
            assert patch_size % 2 == 1
        else:
            assert num_rays is None
            assert patch_size is None

        self.center = torch.zeros(3) if center is None else torch.tensor(center)
        self.depths = None
        self.normals = None


        if split == "trainval":
            _images_train, _train_image_paths, _camtoworlds_train, self.K, _depth_train, _normal_train = _load_renderings(
                root_fp, subject_id, "train"
            )
            _images_val, _test_image_paths, _camtoworlds_val, _, _depth_val, _normal_val = _load_renderings(
                root_fp, subject_id, "val"
            )
            self.images = np.concatenate([_images_train, _images_val])
            self.camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            self.image_paths = [*_train_image_paths, *_test_image_paths]
            
            if _depth_train is not None and _depth_val is not None:
                self.depths = np.concatenate([_depth_train, _depth_val])
            if _normal_train is not None and _normal_val is not None:
                self.normals = np.concatenate([_normal_train, _normal_val])
        else:
            self.images, self.image_paths, self.camtoworlds, self.K, self.depths, self.normals = _load_renderings(
                root_fp, subject_id, split
            )

        self.images = torch.from_numpy(self.images).to(torch.uint8)

        if self.depths is not None:
            self.depths = torch.from_numpy(self.depths)
        if self.normals is not None:
            self.normals = torch.from_numpy(self.normals)

        self.height, self.width = self.images.shape[1:3]
        self.camtoworlds = torch.from_numpy(self.camtoworlds).to(torch.float32)
        self.K = torch.from_numpy(self.K).to(torch.float32)

        # rescale camera positions
        self.camtoworlds[:, :3, 3] = (self.camtoworlds[:, :3, 3] - self.center) * self.scale

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data) -> dict:
        """Process the fetched / cached data with randomness."""
        rgba, rays = data["rgba"], data["rays"]
        pixels, alpha = torch.split(rgba, [3, 1], dim=-1)
        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)
        pixels = pixels * alpha + color_bkgd * (1.0 - alpha)
        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgba", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """Fetch the data (it maybe cached for multiple batches)."""
        num_rays = self.num_rays

        if self.training:
            num_patches = num_rays // (self.patch_size ** 2)
            patch_half_size = self.patch_size // 2

            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_patches,),
                    device=self.images.device,
                )
            else:
                image_id = torch.full((num_patches,), fill_value=index, device=self.images.device)

            

            x = torch.randint(patch_half_size, self.width - patch_half_size, size=(num_patches,), device=self.images.device)
            y = torch.randint(patch_half_size, self.height - patch_half_size, size=(num_patches,), device=self.images.device)

            if patch_half_size > 0:
                offset_xx, offset_yy = torch.meshgrid(
                    torch.arange(-patch_half_size, patch_half_size+1, device=self.images.device),
                    torch.arange(-patch_half_size, patch_half_size+1, device=self.images.device),
                    indexing='xy'
                )
                x = x[:, None, None] + offset_xx[None]
                y = y[:, None, None] + offset_yy[None]
                image_id = torch.repeat_interleave(image_id, repeats=self.patch_size ** 2)
        else:
            image_id = torch.full((self.width * self.height,), fill_value=index, device=self.images.device)
            x, y = torch.meshgrid(
                torch.arange(self.width, device=self.images.device),
                torch.arange(self.height, device=self.images.device),
                indexing="xy",
            )
        x = x.flatten()
        y = y.flatten()
        
        # generate rays
        rgba = self.images[image_id, y, x] / 255.0  # (num_rays, 4)
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5) / self.K[1, 1] * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]
        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.get_radii:
            camera_dirs_cornor = F.pad(
                torch.stack(
                    [
                        (x - self.K[0, 2]) / self.K[0, 0],
                        (y - self.K[1, 2]) / self.K[1, 1] * (-1.0 if self.OPENGL_CAMERA else 1.0),
                    ],
                    dim=-1,
                ),
                (0, 1),
                value=(-1.0 if self.OPENGL_CAMERA else 1.0),
            )  # [num_rays, 3]
            directions_cornor = (
                camera_dirs_cornor[:, None, :] * c2w[:, :3, :3]
            ).sum(dim=-1)
            dx = torch.sqrt(
                torch.sum((directions_cornor - directions) ** 2, -1)
            )
            radii = dx[:, None] * radii_factor
        else:
            radii_value = (
                math.sqrt((0.5 / self.K[0, 0]) ** 2 +
                          (0.5 / self.K[1, 1]) ** 2)
                * radii_factor
            )
            radii = (
                torch.ones(origins.shape[0], 1, device=self.images.device)
                * radii_value
            )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgba = torch.reshape(rgba, (num_rays, 4))
            radii = torch.reshape(radii, (num_rays, 1))
            if self.ndc:
                origins, viewdirs = ndc_rays(
                    self.height, self.width, self.K[0,0],
                    torch.ones(
                        origins.size()[0],
                        dtype=torch.float32,
                        device=origins.device
                    ),
                    origins,
                    viewdirs
                )

        else:
            origins = torch.reshape(origins, (self.width*self.height, 3))
            viewdirs = torch.reshape(viewdirs, (self.width*self.height, 3))
            if self.ndc:
                origins, viewdirs = ndc_rays(
                    self.height, self.width, self.K[0,0],
                    torch.ones(
                        origins.size()[0],
                        dtype=torch.float32,
                        device=origins.device
                    ),
                    origins,
                    viewdirs
                )

            origins = torch.reshape(origins, (self.height, self.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.height, self.width, 3))
            rgba = torch.reshape(rgba, (self.height, self.width, 4))
            radii = torch.reshape(radii, (self.height, self.width, 1))

        xy = torch.stack([x,y], dim=-1)

        rays = Rays(
            origins=origins,
            viewdirs=viewdirs,
            radii=radii if self.get_radii else None,
            cam_idxs=image_id,
            xy=xy
        )

        return {
            "rgba": rgba,  # [h, w, 4] or [num_rays, 4]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }
