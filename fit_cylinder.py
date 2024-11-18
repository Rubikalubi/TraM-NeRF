import argparse
import pyexr
import torch
from pathlib import Path
import json
from tqdm import tqdm
from math import tan
from pytorch3d.structures import Meshes, Pointclouds
import imageio.v2 as imageio
from collections import namedtuple
from skimage.transform import downscale_local_mean

from torchhull import visual_hull

import nvdiffrast.torch as dr

Cylinder = namedtuple("Cylinder", "center direction radius length")

def cylinder_to_mesh(cylinder: Cylinder, resolution: int):
    device = cylinder.center.device

    # generate upper circle
    u = torch.cross(cylinder.direction, torch.tensor([1.0, 0.0, 0.0], device=device), dim=0)
    u = u / torch.linalg.norm(u)

    v = torch.cross(cylinder.direction, u, dim=0)
    v = v / torch.linalg.norm(v)

    angles = torch.linspace(0, 2 * torch.pi, steps=resolution + 1, device=device)[:-1]
    circle_verts = u[None] * cylinder.radius * torch.cos(angles)[:, None]
    circle_verts = circle_verts + v[None] * cylinder.radius * torch.sin(angles)[:, None]
    circle_verts = circle_verts + cylinder.direction[None] * cylinder.length / 2

    # generate vertices as offset of upper circle
    verts = torch.cat([
        circle_verts,
        circle_verts - cylinder.direction * cylinder.length,
        (cylinder.direction * cylinder.length / 2)[None],
        (- cylinder.direction * cylinder.length / 2)[None]
    ], dim=0)

    # move to center
    verts = verts + cylinder.center[None]

    # generate face indices
    top_side_faces = torch.stack([
        torch.arange(resolution), # [0, 1, ..., r - 1]
        torch.arange(1, resolution + 1) % resolution + resolution, # [r + 1, r + 2, ..., r]
        torch.arange(1, resolution + 1) % resolution, # [1, 2, ..., 0]
    ], dim=1)

    bottom_side_faces = torch.stack([
        torch.arange(resolution), # [0, 1, ..., r]
        torch.arange(resolution) + resolution, # [r, r + 1, ..., 2r - 1]
        torch.arange(1, resolution + 1) % resolution + resolution, # [r + 1, r + 2, ..., r]
    ], dim=1)

    top_faces = torch.stack([
        torch.arange(resolution), # [0, 1, ..., r - 1]
        torch.arange(1, resolution + 1) % resolution, # [1, 2, ..., 0]
        torch.ones(resolution) * 2 * resolution, # [2r, 2r, ...]
    ], dim=1)

    bottom_faces = torch.stack([
        torch.arange(resolution) + resolution, # [r, r + 1, ..., 2r - 1]
        torch.ones(resolution) * (2 * resolution + 1), # [2r + 1, 2r + 1, ...]
        torch.arange(1, resolution + 1) % resolution + resolution # [r + 1, r + 2, ..., r]
    ], dim=1)

    faces = torch.cat([
        top_side_faces,
        bottom_side_faces,
        top_faces,
        bottom_faces
    ], dim=0).to(device)

    return verts, faces

def get_opengl_projection_matrix(
    fov_rad: float,
    near: float, far: float,
    cx: float = None, cy: float = None
):
    camera_intrinsics = torch.zeros((4, 4), dtype=torch.float32)
    camera_intrinsics[0, 0] = 1 / tan(fov_rad * 0.5)
    camera_intrinsics[1, 1] = 1 / tan(fov_rad * 0.5)
    
    if cx is not None and cy is not None:
        camera_intrinsics[0, 2] = cx
        camera_intrinsics[1, 2] = cy
        
    camera_intrinsics[2, 2] = -(far + near) / (far - near)
    camera_intrinsics[2, 3] = -(2 * far * near) / (far - near)
    camera_intrinsics[3, 2] = -1.0
    return camera_intrinsics

def render_silhouettes(
    ctx: dr.RasterizeCudaContext,
    verts: torch.Tensor, # (num_verts, 3)
    faces: torch.Tensor, # (num_faces, 3)
    transforms: torch.Tensor, # (num_cams, 4, 4)
    image_size: tuple[int, int]
) -> torch.Tensor: # (num_cams, height, width, 1)
    num_verts = len(verts)

    verts_hom = torch.cat([verts.cuda(), torch.ones(num_verts, 1, device='cuda:0')], dim=1)
    verts_pixel = verts_hom[None] @ transforms.cuda().transpose(1, 2)
    verts_attributes = torch.ones_like(verts_pixel[:, :, :1])
    triangles = faces.to(torch.int32).cuda()

    rast, diff_rast = dr.rasterize(ctx, verts_pixel, triangles, resolution=image_size)
    rendered_masks, _ = dr.interpolate(verts_attributes, rast, triangles, rast_db=diff_rast, diff_attrs=None)
    rendered_masks = dr.antialias(rendered_masks, rast, verts_pixel, triangles)
    rendered_masks = torch.flip(rendered_masks, dims=(1,))
    return rendered_masks

def cartesian_to_spherical(v):
        theta = torch.arccos(v[..., 2] / torch.linalg.norm(v, dim=-1))
        phi = torch.sign(v[..., 1]) * torch.arccos(v[..., 0] / torch.linalg.norm(v[..., :2], dim=-1))
        return theta, phi

def spherical_to_cartesian(theta, phi):
    return torch.stack([
        torch.sin(theta) * torch.cos(phi),
        torch.sin(theta) * torch.sin(phi),
        torch.cos(theta)
    ], dim=-1)


def fit_cylinder(
        scene_dir: Path,
        masks_dir: Path,
        has_exr: bool = False,
        masks_ds_factor: int = 1,
        num_views: int = 16
):
    # get transforms file

    with (scene_dir / "transforms_train.json").open() as transforms_file:
        transforms_json = json.load(transforms_file)

    # build intrinsic matrix

    if "cx" in transforms_json:
        h, w, _ = imageio.imread(scene_dir / "train" / Path(transforms_json["frames"][0]["file_path"]).name).shape
        near, far = 0.001, 1000
        K = torch.tensor(transforms_json["intrinsics"])
        K = torch.tensor([
            [2 * K[0,0] / w, -2 * K[0,1] / w,       (w - 2 * K[0,2]) / w,                              0],
            [             0,  2 * K[1,1] / h,      (-h + 2 * K[1,2]) / h,                              0],
            [             0,               0, (-far - near)/(far - near), -2 * far * near / (far - near)],
            [             0,               0,                         -1,                              0]
        ])

    else:
        camera_angle_x = float(transforms_json["camera_angle_x"])
        K = get_opengl_projection_matrix(camera_angle_x, 0.001, 1000)
    
    # K =  @ K
    # K = K @ torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0]))


    # read masks
    
    masks = []
    images = []
    transforms = []
    cam_to_worlds = []

    for frame in tqdm(transforms_json["frames"], desc='load masks'):
        if has_exr:
            with pyexr.open(str(scene_dir / (frame["file_path"] + "_mask.exr"))) as exr_file:
                mask = torch.from_numpy(exr_file.get("ViewLayer.Emit"))
                mask = mask.to(torch.float32) / 255
            with pyexr.open(str(scene_dir / (frame["file_path"] + ".exr"))) as exr_file:
                image = pyexr.tonemap(exr_file.get("Composite.Combined"))
                image = torch.from_numpy(image)
        else:
            mask_path = masks_dir / (Path(frame["file_path"]).name + ".png")
            
            if mask_path.exists():
                mask = imageio.imread(mask_path)
                
                if len(mask.shape) == 3:
                    mask = mask[..., :3]
                elif len(mask.shape) == 2:
                    mask = mask[..., None]
                
                mask = torch.from_numpy(mask)
                mask = mask.to(torch.float32) / 255
                
                image = torch.from_numpy(imageio.imread(scene_dir / frame["file_path"]))
                image = image.to(torch.float32) / 255
            else:
                continue
        
        if masks_ds_factor != 1:
            mask = downscale_local_mean(mask.numpy(), factors=(masks_ds_factor, masks_ds_factor, 1))
            mask = torch.from_numpy(mask)

        mask = torch.linalg.norm(mask, dim=-1, keepdim=True)
        mask = (mask / mask).nan_to_num(0)
        masks.append(mask)
        images.append(image)

        cam_to_world = torch.tensor(frame["transform_matrix"])
        transform = K @ cam_to_world.inverse()
        transforms.append(transform)
        cam_to_worlds.append(cam_to_world)

    masks = torch.stack(masks)
    images = torch.stack(images)
    transforms = torch.stack(transforms)
    cam_to_worlds = torch.stack(cam_to_worlds)

    transforms_torchhull = torch.diag(torch.tensor([1.0, -1.0, 1.0, 1.0]))[None] @ transforms

    print("Found", len(masks), "masks")

    num_views = min(num_views, len(masks))
    view_idxs = torch.randperm(len(masks))[:num_views]
    initial_scale = 20

    # search for object in large area

    verts, faces = visual_hull(
        masks=masks[view_idxs].cuda(),
        transforms=transforms_torchhull[view_idxs].cuda(),
        level=8,
        cube_corner_bfl=(-initial_scale / 2,) * 3,
        cube_length=initial_scale,
        masks_partial=False,
        unique_verts=True
    )

    # use bounding box to optimize region

    mesh_center = verts.mean(dim=0) * initial_scale
    max_distance = torch.linalg.norm(verts * initial_scale - mesh_center[None], ord=1, dim=1).max()

    verts, faces = visual_hull(
        masks=masks[view_idxs].cuda(),
        transforms=transforms_torchhull[view_idxs].cuda(),
        level=8,
        cube_corner_bfl=(-max_distance / 2,) * 3,
        cube_length=max_distance,
        masks_partial=False,
        unique_verts=True
    )

    mesh = Meshes(verts[None], faces[None])
    normals = mesh.verts_normals_list()

    # find initial axis estimate

    pointcloud = Pointclouds(verts[None]).cuda()
    neighborhood_size = 50
    smooth_normals = pointcloud.estimate_normals(neighborhood_size=neighborhood_size).cpu()
    theta, phi = cartesian_to_spherical(smooth_normals[0])

    sph_coords = torch.stack([theta, phi], dim=-1)
    counts, bin_edges = torch.histogramdd(sph_coords.cpu(), bins=50, range=[0, torch.pi, -torch.pi, torch.pi])

    idx = (counts == torch.max(counts)).nonzero()[0]
    max_theta = bin_edges[0][idx[0]]
    max_phi = bin_edges[1][idx[1]]
    max_normal = spherical_to_cartesian(max_theta, max_phi).cuda()

    v1 = torch.cross(max_normal, torch.tensor([1.0, 0.0, 0.0], device='cuda:0'), dim=0)
    v1 /= torch.linalg.norm(v1)
    v2 = torch.cross(v1, max_normal, dim=0)
    v2 /= torch.linalg.norm(v2)
    projected_verts = torch.cat([verts @ v1[:, None], verts @ v2[:, None]], dim=-1)

    center = projected_verts.mean(dim=0)
    center_dists = torch.linalg.norm(projected_verts - center[None], dim=1)

    filtered_verts = verts

    axis_sph = torch.tensor([max_theta, max_phi], requires_grad=True)

    optimizer = torch.optim.Adam([axis_sph], lr=0.01)
    all_dist_vars = []
    num_bins = 100
    num_iterations = 1000

    for i in tqdm(range(num_iterations + 1), desc="axis opt"):
        optimizer.zero_grad()

        cylinder_axis = spherical_to_cartesian(*axis_sph).cuda()

        v1 = torch.cross(cylinder_axis, torch.tensor([1.0, 0.0, 0.0], device='cuda:0'), dim=0)
        v1 = v1 / torch.linalg.norm(v1)
        v2 = torch.cross(v1, cylinder_axis, dim=0)
        v2 = v2 / torch.linalg.norm(v2)

        projected_verts = torch.cat([filtered_verts @ v1[:, None], filtered_verts @ v2[:, None]], dim=-1)
        center = projected_verts.mean(dim=0)

        center_dists = torch.linalg.norm(projected_verts - center[None], dim=1)
        discrete_center_dists = (center_dists - center_dists.min()) / (center_dists.max() - center_dists.min()) * num_bins
        discrete_center_dists = discrete_center_dists.round()
        dist_mode_idx = discrete_center_dists.mode().values
        dist_mode = (center_dists.max() - center_dists.min()) * dist_mode_idx / num_bins + center_dists.min()
        

        dist_var = ((center_dists - dist_mode) ** 2).mean()
        dist_var.backward()
        all_dist_vars.append(dist_var.item())
        optimizer.step()

    # get initial cylinder parameters from bounding box

    normals_dot_axis = (normals[0] @ cylinder_axis[:, None])[:, 0]
    threshold = 0.3
    top_mask = normals_dot_axis > threshold
    bottom_mask = normals_dot_axis < -threshold

    cylinder_center = verts.mean(dim=0)
    top_proj = (verts[top_mask] - cylinder_center).mean(dim=0) @ cylinder_axis
    top = cylinder_center + cylinder_axis * top_proj
    top = top.detach().cpu()

    bottom_proj = (verts[bottom_mask] - cylinder_center).mean(dim=0) @ cylinder_axis
    bottom = cylinder_center + cylinder_axis * bottom_proj
    bottom = bottom.detach().cpu()

    length = torch.abs(top_proj - bottom_proj).detach().cpu()
    radius = dist_mode.detach().cpu()
    cylinder_center = cylinder_center.detach().cpu()

    cylinder = Cylinder(
        center=cylinder_center.detach().cpu(),
        direction=cylinder_axis.detach().cpu(),
        radius=radius.detach().cpu(),
        length=length.detach().cpu()
    )

    verts_center = verts.mean(dim=0)
    centered_verts = verts - verts_center
    U, S, V_t = torch.linalg.svd(centered_verts, full_matrices=False)

    rotated_verts = U @ torch.diag(S)
    rotated_bb_min = rotated_verts.amin(dim=0)
    rotated_bb_max = rotated_verts.amax(dim=0)

    rotated_bb_vis = torch.tensor([
        [rotated_bb_min[0], rotated_bb_min[1], rotated_bb_min[2]],
        [rotated_bb_min[0], rotated_bb_min[1], rotated_bb_max[2]],
        [rotated_bb_min[0], rotated_bb_max[1], rotated_bb_min[2]],
        [rotated_bb_min[0], rotated_bb_max[1], rotated_bb_max[2]],
        [rotated_bb_max[0], rotated_bb_min[1], rotated_bb_min[2]],
        [rotated_bb_max[0], rotated_bb_min[1], rotated_bb_max[2]],
        [rotated_bb_max[0], rotated_bb_max[1], rotated_bb_min[2]],
        [rotated_bb_max[0], rotated_bb_max[1], rotated_bb_max[2]],
    ]).cuda()

    bb_vis = rotated_bb_vis @ V_t + verts_center
    rotated_bb_vis += verts_center

    cylinder = Cylinder(
        center=bb_vis.mean(dim=0),
        direction=V_t[0],
        radius=(rotated_bb_max[1] - rotated_bb_min[1] + rotated_bb_max[2] - rotated_bb_min[2]) / 4,
        length=(rotated_bb_max[0] - rotated_bb_min[0])
    )

    # rotated_verts += verts_center

    ctx = dr.RasterizeGLContext()

    opt_cylinder = Cylinder(
        center=cylinder.center.clone().cuda(),
        direction=cylinder.direction.clone().cuda(),
        radius=cylinder.radius.clone().cuda(),
        length=cylinder.length.clone().cuda()
    )

    opt_cylinder.center.requires_grad = True
    opt_cylinder.direction.requires_grad = True
    opt_cylinder.radius.requires_grad = True
    opt_cylinder.length.requires_grad = True

    lr = 0.01
    optimizer = torch.optim.SGD([*opt_cylinder], lr=lr)
    masks_gpu = masks.cuda()

    all_losses = []

    min_num_iterations = 1_000
    num_iterations = 10_000 + 1

    early_stopping_window = 2000
    early_stopping_num = 10_000

    best_loss = 1e10
    best_i = -1

    with tqdm(range(num_iterations), desc="silhouette opt") as pbar:
        for i in pbar:
            optimizer.zero_grad()

            # just to visualize the initial state below
            if i == 0:
                optimizer.lr = 0.0
            else:
                optimizer.lr = lr

            opt_verts, opt_faces = cylinder_to_mesh(opt_cylinder, resolution=200)
            rendered_masks = render_silhouettes(ctx, opt_verts, opt_faces, transforms, masks.shape[1:3])

            loss = ((rendered_masks - masks_gpu) ** 2).mean()
            loss.backward()
            optimizer.step()

            # normalize direction
            with torch.no_grad():
                opt_cylinder.direction[:] = opt_cylinder.direction / torch.linalg.norm(opt_cylinder.direction)
            all_losses.append(loss.item())

            if all_losses[-1] < best_loss or (best_i + early_stopping_window) <= i:
                best_loss = all_losses[-1]
                best_i = i

            pbar.set_postfix(dict(loss=loss.item()), best=best_loss, best_i=best_i)

            if i > min_num_iterations and i - best_i > early_stopping_num:
                print("early stopping after", i, "iterations")
                break

    p_origin = (opt_cylinder.center - 0.5 * opt_cylinder.length * opt_cylinder.direction).detach()
    p_end = (opt_cylinder.center + 0.5 * opt_cylinder.length * opt_cylinder.direction).detach()

    # print resulting parameters

    print("\n\nResulting cylinder parameters:\n")

    print(json.dumps([
        {
            "type": "cylinder",
            "origin": str(p_origin.tolist()),
            "end": str(p_end.tolist()),
            "radius": opt_cylinder.radius.item()
        }
    ], indent=4))

    print("\n\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scene_dir",
        type=Path,
        help="Path to the scene dir."
    )
    parser.add_argument(
        "--masks_dir",
        type=Path,
        help=(
            "Path to a directory containing masks for a subset of the train "
            "images. Masks have to be named <name_of_image>.png (including "
            "the original suffix)."
        )
    )
    parser.add_argument(
        "--has_exr",
        action="store_true",
        help=(
            "Set this flag if masks are given as a layer in the original train "
            "data."
        )
    )
    parser.add_argument(
        "--masks_ds_factor",
        type=int,
        default=1,
        help="Downscale factor for the mask images."
    )
    parser.add_argument(
        "--num_views",
        type=int,
        default=16,
        help=(
            "Number of masks that are randomly drawn from <masks_dir> to fit the "
            "parameters to."
        )
    )
    args = parser.parse_args()

    fit_cylinder(**args.__dict__)

if __name__ == "__main__":
    main()
