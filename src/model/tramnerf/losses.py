import warnings
import torch
from typing import Any, Protocol
from src.data.utils import Rays
from einops import rearrange
from src.model.helper import FunctionRegistry
import gin
import torch.nn.functional as F
import numpy as np


class LossFunction(Protocol):
    def __call__(
            self,
            rays: Rays,
            rendered_results: dict[torch.Tensor]
        ) -> tuple[float, dict[torch.Tensor]]:
        pass

def is_coarse_key(key: str):
    return "coarse_" in key

def is_fine_key(key: str):
    return "fine_" in key

def is_shared_key(key: str):
    return not (is_coarse_key(key) or is_fine_key(key))

def patch_outputs(loss, debug_tensors, prefix: str):
    if debug_tensors is None:
        debug_tensors = dict()
    debug_tensors = {f"{prefix}{k}": v for k, v in debug_tensors.items()}    
    return loss, debug_tensors

class LossFunctionRegistry(FunctionRegistry[LossFunction]):
    def __init__(self) -> None:
        super().__init__()
        self.render_fn = None

    def set_render_fn(self, fn):
        print("render fn set")
        self.render_fn = fn

    def register_coarse_fine(self, name: str | None = None):
        def _register_coarse_fine(fn: LossFunction) -> LossFunction:
            _name = fn.__name__ if name is None else name

            def coarse_loss(rays, rendered_results):
                coarse_outputs = {k.replace("coarse_", ""): v for k, v in rendered_results.items() if is_coarse_key(k)}
                shared_outputs = {k: v for k, v in rendered_results.items() if is_shared_key(k)}
                loss, debug_tensors = fn(rays=rays, rendered_results={**coarse_outputs, **shared_outputs})
                return patch_outputs(loss, debug_tensors, prefix="coarse_")
            super(LossFunctionRegistry, self).register(f"coarse_{_name}")(coarse_loss)

            def fine_loss(rays, rendered_results):
                fine_outputs = {k.replace("fine_", ""): v for k, v in rendered_results.items() if is_fine_key(k)}
                shared_outputs = {k: v for k, v in rendered_results.items() if is_shared_key(k)}
                loss, debug_tensors = fn(rays=rays, rendered_results={**fine_outputs, **shared_outputs})
                return patch_outputs(loss, debug_tensors, prefix="fine_")
            super(LossFunctionRegistry, self).register(f"fine_{_name}")(fine_loss)

            return fn
        
        return _register_coarse_fine
    
    def register_rerendering_loss(self, name: str | None = None):
        def _register_rerendering_loss(fn) -> LossFunction:
            _name = fn.__name__ if name is None else name
            def rerender_loss(rays, rendered_results):
                return fn(rays=rays, rendered_results=rendered_results, render_fn=self.render_fn)
            super(LossFunctionRegistry, self).register(_name)(rerender_loss)
        return _register_rerendering_loss

    
def acc_reduce(x, y, reduce):
    if reduce == 'or':
        return x | y
    elif reduce == 'min':
        return torch.minimum(x, y)
    elif reduce == 'max':
        return torch.maximum(x, y)
    elif reduce == 'sum':
        return x + y
    else:
        ValueError(f"Unknown reduction '{reduce}'")


def get_accumulated_mirror_result(rendered_results: dict[str, torch.Tensor], key: str = 'valid', prefix: str = 'coarse_', reduce='or', check_valid: bool = False, default: Any = 0.0):
    i = 0
    result = None

    while f"{prefix}mirror_{i}_bounce_0_{key}" in rendered_results:
        cur_result = rendered_results[f"{prefix}mirror_{i}_bounce_0_{key}"]
        if check_valid:
            cur_valid = rendered_results[f"{prefix}mirror_{i}_bounce_0_valid"]
            if result is None:
                result = cur_result.clone()
                result[~cur_valid] = default
            else:
                result[cur_valid] = acc_reduce(result[cur_valid], cur_result[cur_valid], reduce)
        else:
            result = cur_result if result is None else acc_reduce(result, cur_result, reduce)
        i += 1

    return result
    
LOSS_FUNCTIONS = LossFunctionRegistry()

@LOSS_FUNCTIONS.register_coarse_fine()
def forward_facing_normals_loss(rays: Rays, rendered_results: dict):
    warnings.warn(RuntimeWarning("Current only considering first bounce for mirror loss."))
    loss = (rays.viewdirs[:, None, :] * rendered_results["mirror_0_bounce_0_pred_normals"][:, :, None])[:, 0, 0]
    return torch.relu(loss).mean(), None

@LOSS_FUNCTIONS.register_coarse_fine("l2_rendering_loss")
def l2_loss(
        rays: Rays,
        rendered_results: dict[torch.Tensor],
    ) -> tuple[float, dict[torch.Tensor]]:
    squared_error = (rendered_results["rgb"] - rendered_results["ground_truth"]) ** 2
    return squared_error.mean(), None

@LOSS_FUNCTIONS.register_coarse_fine("l1_rendering_loss")
def l2_loss(
        rays: Rays,
        rendered_results: dict[torch.Tensor],
    ) -> tuple[float, dict[torch.Tensor]]:
    abs_error = (rendered_results["rgb"] - rendered_results["ground_truth"]).abs()
    return abs_error.mean(), None

@LOSS_FUNCTIONS.register_coarse_fine("huber_rendering_loss")
@gin.configurable
def huber_loss(
        rays: Rays,
        rendered_results: dict[torch.Tensor],
        delta: float = 0.5
    ) -> tuple[float, dict[torch.Tensor]]:
    error = F.huber_loss(rendered_results["rgb"], rendered_results["ground_truth"], delta=delta)
    return error.mean(), None

@LOSS_FUNCTIONS.register_coarse_fine("scheduled_lp_rendering_loss")
@gin.configurable
def scheduled_lp_loss(
        rays: Rays,
        rendered_results: dict[torch.Tensor],
        p_values: list[float] = [2.0, 1.0, 2.0, 2.0],
        steps: list[int] = [0, 2000, 100_000, 1e10]
    ) -> tuple[float, dict[torch.Tensor]]:
    x = rendered_results["step"].item()
    idx = np.searchsorted(steps, x)
    a, b = steps[idx - 1], steps[idx]
    x = (x - a) / (b - a)
    p = p_values[idx - 1] * (1 - x) + p_values[idx] * x
    error = (rendered_results["rgb"] - rendered_results["ground_truth"]).norm(dim=1, p=p) ** p
    return error.mean(), {"scheduled_p": torch.tensor([p])}


def rotation_matrices_angle(rot_1: torch.Tensor, rot_2: torch.Tensor):
    rot = rot_1 @ rot_2.transpose(-1, -2)
    trace = rot.diagonal(offset=0, dim1=-2, dim2=-1).sum(dim=-1)
    return torch.arccos((trace - 1) / 2).nan_to_num(0.0)

def extrinsics_distances(ctw_1: torch.Tensor, ctw_2: torch.Tensor):
    rotation_angle = rotation_matrices_angle(ctw_1[..., :3, :3], ctw_2[..., :3, :3])
    translation_distance = (ctw_1[..., :3, 3] - ctw_2[..., :3, 3]).norm(dim=-1)
    return torch.abs(rotation_angle) * translation_distance / torch.pi


@LOSS_FUNCTIONS.register_coarse_fine()
def reprojection_loss(
        rays: Rays,
        rendered_results: dict[torch.Tensor]
) -> tuple[float, dict[torch.Tensor]]:
    height, width = rendered_results["train_images"].shape[1:3]
    world_to_cams = rendered_results["train_cam_to_worlds"].inverse()
    channel_first_images = rearrange(
        rendered_results["train_images"][:, :, :, :3] / 255,
        "n h w c -> n c h w"
    )
    pairwise_cam_dists = extrinsics_distances(
        rendered_results["train_cam_to_worlds"][None, :],
        rendered_results["train_cam_to_worlds"][:, None]
    )

    # compute 3D locations of pixels
    x_world = rays.origins + rays.viewdirs * rendered_results["dist"][:, None]
    x_world = torch.cat([x_world, torch.ones_like(x_world[:, :1])], dim=1) # (n, 4)

    # sample nearby camera
    cam_sampling_probs = (pairwise_cam_dists.amax(dim=1) - pairwise_cam_dists).nan_to_num(0.0)
    nearby_cam_idxs = torch.multinomial(cam_sampling_probs[rays.cam_idxs], num_samples=1)[:, 0]

    # project into images
    world_to_cam = world_to_cams[nearby_cam_idxs] # (m, 4, 4)
    x_cam = (world_to_cam @ x_world[:, :, None])[:, :, 0]
    x_cam = x_cam[:, :3] / x_cam[:, 2:3]
    x_screen = (rendered_results["train_intrinsics"][None] @ x_cam[:, :, None])[:, :, 0]
    x_screen[:, 0] = -(2 * x_screen[:, 0] / width - 1)
    x_screen[:, 1] = 2 * x_screen[:, 1] / height - 1  # (n, 3)

    # sample colors from nearby camera
    sampled_rgb = torch.nn.functional.grid_sample(
        input=channel_first_images[nearby_cam_idxs],
        grid=x_screen[:, None, None, :2]
    )[:, :, 0, 0] # (n, 3)

    # mask regions where samples were outside of frustum
    with torch.no_grad():
        sample_outside_img = (x_screen.abs() > 1.0).any(dim=1)
        masked_rgb = rendered_results["ground_truth"].clone()
        masked_rgb[sample_outside_img] = 0.0

    # compute loss
    loss = torch.mean((sampled_rgb - masked_rgb) ** 2)

    # generate debug output
    debug_tensors = {
        f"dist_reprojected": sampled_rgb.detach()
    }
    return loss, debug_tensors

@LOSS_FUNCTIONS.register_coarse_fine()
def weighted_reprojection_loss(
        rays: Rays,
        rendered_results: dict[torch.Tensor]
) -> tuple[float, dict[torch.Tensor]]:
    height, width = rendered_results["train_images"].shape[1:3]
    world_to_cams = rendered_results["train_cam_to_worlds"].inverse()
    channel_first_images = rearrange(
        rendered_results["train_images"][:, :, :, :3] / 255,
        "n h w c -> n c h w"
    )
    pairwise_cam_dists = extrinsics_distances(
        rendered_results["train_cam_to_worlds"][None, :],
        rendered_results["train_cam_to_worlds"][:, None]
    )

    # compute 3D locations of pixels
    x_world = rays.origins + rays.viewdirs * rendered_results["dist"][:, None]
    x_world = torch.cat([x_world, torch.ones_like(x_world[:, :1])], dim=1) # (n, 4)

    # sample nearby camera
    max_cam_dist = pairwise_cam_dists.amax()
    cam_sampling_probs = (max_cam_dist - pairwise_cam_dists).nan_to_num(0.0) / max_cam_dist
    # nearby_cam_idxs = torch.multinomial(cam_sampling_probs[rays.cam_idxs], num_samples=1)[:, 0]
    sampled_cam_idxs = torch.randint(0, len(channel_first_images), (len(rays.cam_idxs),))
    cam_weights = cam_sampling_probs[rays.cam_idxs, sampled_cam_idxs]

    # project into images
    world_to_cam = world_to_cams[sampled_cam_idxs] # (m, 4, 4)
    x_cam = (world_to_cam @ x_world[:, :, None])[:, :, 0]
    x_cam = x_cam[:, :3] / x_cam[:, 2:3]
    x_screen = (rendered_results["train_intrinsics"][None] @ x_cam[:, :, None])[:, :, 0]
    x_screen[:, 0] = -(2 * x_screen[:, 0] / width - 1)
    x_screen[:, 1] = 2 * x_screen[:, 1] / height - 1  # (n, 3)

    # sample colors from nearby camera
    sampled_rgb = torch.nn.functional.grid_sample(
        input=channel_first_images[sampled_cam_idxs],
        grid=x_screen[:, None, None, :2]
    )[:, :, 0, 0] # (n, 3)

    # mask regions where samples were outside of frustum
    sample_outside_img = (x_screen.abs() > 1.0).any(dim=1)
    cam_weights[sample_outside_img] = 0.0

    # mask regions with mirrors
    is_mirror = get_accumulated_mirror_result(rendered_results, key='valid', prefix='', reduce='or')
    if is_mirror is not None:
        cam_weights[is_mirror] = 0.0

    # compute loss
    loss = torch.mean(cam_weights[:, None] * (sampled_rgb - rendered_results["ground_truth"]) ** 2)

    # generate debug output
    debug_tensors = {
        f"dist_reprojected": cam_weights[:, None] * sampled_rgb.detach()
    }
    return loss, debug_tensors

@LOSS_FUNCTIONS.register_rerendering_loss()
def weighted_depth_reprojection_loss(
        rays: Rays,
        rendered_results: dict[torch.Tensor],
        render_fn,
        occlusion_threshold: float = 0.1,
) -> tuple[float, dict[torch.Tensor]]:
    num_train_images = len(rendered_results["train_images"])
    num_batch = len(rays.origins)
    height, width = rendered_results["train_images"].shape[1:3]
    world_to_cams = rendered_results["train_cam_to_worlds"].inverse()
    pairwise_cam_dists = extrinsics_distances(
        rendered_results["train_cam_to_worlds"][None, :],
        rendered_results["train_cam_to_worlds"][:, None]
    )

    # compute 3D locations of pixels
    x_world = rays.origins + rays.viewdirs * rendered_results["fine_dist"][:, None]
    x_world = torch.cat([x_world, torch.ones_like(x_world[:, :1])], dim=1) # (n, 4)

    # sample nearby camera
    max_cam_dist = pairwise_cam_dists.amax()
    cam_sampling_probs = (max_cam_dist - pairwise_cam_dists).nan_to_num(0.0) / (max_cam_dist + 1e-8)
    # nearby_cam_idxs = torch.multinomial(cam_sampling_probs[rays.cam_idxs], num_samples=1)[:, 0]
    sampled_cam_idxs = torch.randint(0, num_train_images, (len(rays.cam_idxs),))
    cam_weights = cam_sampling_probs[rays.cam_idxs, sampled_cam_idxs]

    # project into images
    world_to_cam = world_to_cams[sampled_cam_idxs] # (m, 4, 4)
    x_cam = (world_to_cam @ x_world[:, :, None])[:, :, 0]
    x_cam = x_cam[:, :3] / (x_cam[:, 2:3] + 1e-8)
    x_screen = (rendered_results["train_intrinsics"][None] @ x_cam[:, :, None])[:, :, 0]
    x_screen[:, 0] = -(2 * x_screen[:, 0] / width - 1)
    x_screen[:, 1] = 2 * x_screen[:, 1] / height - 1  # (n, 3)

    sampled_origins = rendered_results["train_cam_to_worlds"][sampled_cam_idxs, :3, 3]
    sampled_viewdirs = x_world[:, :3] - sampled_origins
    sampled_viewdirs = sampled_viewdirs / (sampled_viewdirs.norm(dim=-1, keepdim=True) + 1e-8)

    new_rays = Rays(
        origins=sampled_origins,
        viewdirs=sampled_viewdirs,
        radii=rays.radii,
        cam_idxs=sampled_cam_idxs,
        xy=None
    )
    new_rendered_results = render_fn(new_rays)
    new_x_world = new_rays.origins + new_rays.viewdirs * new_rendered_results["fine_dist"][:, None]

    # mask regions where samples were outside of frustum
    sample_outside_img = (x_screen.abs() > 1.0).any(dim=1)
    cam_weights[sample_outside_img] = 0.0

    # compute loss
    dist_to_new_cam = (x_world[:, :3] - sampled_origins).norm(dim=-1)

    is_occluded = (dist_to_new_cam - new_rendered_results["fine_dist"].detach()) > occlusion_threshold
    depth_reproj_error = cam_weights * (new_x_world - x_world[:, :3]).norm(dim=-1)
    depth_reproj_error[is_occluded] = 0.0

    # generate debug output
    debug_tensors = {
        f"depth_reprojection_error": depth_reproj_error.detach()
    }

    return depth_reproj_error.mean(), debug_tensors
