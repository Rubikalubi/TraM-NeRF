from collections import defaultdict
from dataclasses import dataclass, field
from math import floor
from pathlib import Path
from typing import Callable, NamedTuple
import warnings
import tqdm

import gin
from time import time
import torch

from src.data.utils import Rays
from src.model.helper import clamp_cast_uint8, depth_to_rgb, directions_to_rgb
from src.model.tramnerf.losses import LOSS_FUNCTIONS


class TrainProgressBar:
    def __init__(self, num_train_it: int, batch_size: int, num_eval_it: int, eval_size: int, first_it: int):
        self.num_train_it = num_train_it
        self.batch_size = batch_size
        self.num_eval_it = num_eval_it
        self.eval_size = eval_size

        self.num_at_eval_start = 0
        self.state = "train"
        self.cur_it = first_it
        self.last_step_time = time()

        self.psnr = 0.0

        total = num_train_it * batch_size + num_eval_it * eval_size
        init_fraction = (first_it - 1) / num_train_it
        init = first_it * batch_size + floor(num_eval_it * init_fraction) * eval_size
        self.pbar = tqdm.tqdm(total=total, unit_scale=True, unit=" rays", smoothing=0.15, initial=init)

    def set_state(self, new_state: str):
        if new_state == "eval-test" or new_state == "eval-train":
            self.num_at_eval_start = self.pbar.n
        elif new_state == "train":
            pass
        else:
            raise ValueError(f"Unknown state '{new_state}'")
        self.state = new_state
        self.update_postfix()

    def update_postfix(self):
        duration = time() - self.last_step_time

        if duration < 1.0:
            dps_str = f"{1 / duration:.2f} it/s"
        else:
            dps_str = f"{duration:.2f} s/it"

        pfs = f"it {self.cur_it}/{self.num_train_it} ({dps_str}), psnr={self.psnr:.2f}, {self.state}"

        if self.state.startswith("eval-"):
            num_since_eval_start = self.pbar.n - self.num_at_eval_start
            num_it_for_eval = self.eval_size // self.batch_size
            remaining_eval_it = (self.eval_size - num_since_eval_start) // self.batch_size
            eta = remaining_eval_it * duration
            pfs += f" {num_it_for_eval - remaining_eval_it}/{num_it_for_eval} (ETA: {eta:.0f}s)"

        self.pbar.set_postfix_str(pfs)


    def step(self, size: int = None, psnr: float = None):
        if psnr is not None:
            self.psnr = psnr

        if self.state == "train":
            self.cur_it += 1

        self.pbar.update(self.batch_size if size is None else size)
        self.update_postfix()
        self.last_step_time = time()


@gin.configurable
@dataclass
class Config():
    seed: int = 42
    scene_name: str = ""
    model: str = "tramnerf"
    scale: float = 1.0
    center: list[float] = None
    farplane: float = 25.0  # after scaling
    nearplane: float = 1.0  # after scaling
    test_image_id: int = 0
    iterations: int = 30000
    eval_every: int = 200
    lr: float = 0.001
    bsz: int = 2**14
    patch_size: int = 1
    use_gradient_clipping: bool = False
    white_background: bool = True

    reduce_on_plateau_factor: float = None
    reduce_on_plateau_patience: int = None
    reduce_on_plateau_cooldown: int = None
    reduce_on_plateau_min_lr: float = None

    use_mc_model: bool =  False

    render_train_view_for_eval: bool = False
    train_eval_image_id: int = None

    loss_weights: dict[str, float] = field(default_factory=lambda: {
        "coarse_l2_rendering_loss": 0.1,
        "fine_l2_rendering_loss": 1.0
    })

    check_for_nan_grads: bool = True
    check_for_nan_loss: bool = True
    custom_checkpoint_at: list[int] = field(default_factory=list)


class WriterInfo(NamedTuple):
   to_image_fn: Callable[[torch.Tensor], torch.Tensor]
   only_once: bool = False

def setup_writers(config: Config, num_mirrors: int, num_bounces: int):
    default_to_tb = WriterInfo(clamp_cast_uint8, only_once=False)
    default_to_tb_only_once = WriterInfo(clamp_cast_uint8, only_once=True)
    depth_to_tb = WriterInfo(lambda t: clamp_cast_uint8(depth_to_rgb(t, config.nearplane, config.farplane)), only_once=False)
    directions_to_tb = WriterInfo(lambda t: clamp_cast_uint8(directions_to_rgb(t)), only_once=False)
    directions_to_tb_choose_once = WriterInfo(directions_to_tb.to_image_fn, only_once=True)

    OUTPUT_TO_IMAGE_DICT = defaultdict(lambda: default_to_tb)

    for level_key in ("coarse", "fine"):
        # set non-default writers for each mirror
        for i in range(num_mirrors):
            for j in range(num_bounces):
                key_base = f"{level_key}_mirror_{i}_bounce_{j}"
                OUTPUT_TO_IMAGE_DICT[f"{key_base}_t"] = depth_to_tb
                OUTPUT_TO_IMAGE_DICT[f"{key_base}_pred_normals"] = directions_to_tb
                OUTPUT_TO_IMAGE_DICT[f"{key_base}_normal"] = directions_to_tb

        # other level-dependent outputs
        OUTPUT_TO_IMAGE_DICT[f"{level_key}_dist"] = depth_to_tb
        OUTPUT_TO_IMAGE_DICT[f"{level_key}_final_viewdirs"] = directions_to_tb_choose_once
        OUTPUT_TO_IMAGE_DICT[f"{level_key}_first_reflection_direction"] = directions_to_tb_choose_once
        OUTPUT_TO_IMAGE_DICT[f"{level_key}_ray_num_bounces"] = WriterInfo(lambda t: clamp_cast_uint8(t, maximum=num_bounces), only_once=True)

    # level-independent outputs
    OUTPUT_TO_IMAGE_DICT["ground_truth_train"] = default_to_tb_only_once
    OUTPUT_TO_IMAGE_DICT["ground_truth"] = default_to_tb_only_once

    return OUTPUT_TO_IMAGE_DICT


def render_full_image(
        model,
        dataset_sample,
        loss_data_fn,
        render_keys=('fine_rgb',),
        chunk_size=None,
        return_debug_images: bool = True,
        return_loss_outputs: bool = False,
        downsample_step: int = 1,
        pbar: TrainProgressBar = None
):
    config = Config()
    rays = dataset_sample["rays"]

    final_shape = rays.origins[::downsample_step, ::downsample_step].shape

    origins = rays.origins[::downsample_step, ::downsample_step].reshape(-1, 3)
    directions = rays.viewdirs[::downsample_step, ::downsample_step].reshape(-1, 3)
    directions = directions / torch.linalg.norm(directions, dim=-1, keepdim=True)

    if rays.radii is not None:
        radii = rays.radii[::downsample_step, ::downsample_step].reshape(-1, 1)
    pass_radii = rays.radii is not None

    cam_idxs = rays.cam_idxs[::(downsample_step **2)]
    xy = rays.xy[::downsample_step, ::downsample_step]
    pixels = dataset_sample["pixels"][::downsample_step, ::downsample_step].reshape(-1, 3)

    if render_keys is not None:
        output_batches = {k: [] for k in render_keys}
    else:
        output_batches = defaultdict(list)

    last_chunk_size = final_shape[0] * final_shape[1] % chunk_size

    for i in range(0, final_shape[0] * final_shape[1], chunk_size):
        rays_batch = Rays(
            origins=origins[i:i+chunk_size],
            viewdirs=directions[i:i+chunk_size],
            radii=radii[i:i+chunk_size] if pass_radii else None,
            cam_idxs=cam_idxs[i:i+chunk_size],
            xy=xy[i:i+chunk_size]
        )
        sample_batch = {
            'rays': rays_batch,
            'pixels': pixels[i:i+chunk_size]
        }

        rendered_results = model(
            rays=rays_batch,
            randomized=False,
            white_bkgd=config.white_background,
            near=config.nearplane,
            far=config.farplane,
            return_debug_images=return_debug_images
        )
        rendered_results = {**rendered_results, **loss_data_fn(sample_batch)}

        if return_loss_outputs:
            # compute primary losses

            loss_params = dict(rendered_results=rendered_results, rays=rays_batch)
            loss_dict = {name: LOSS_FUNCTIONS[name](**loss_params) for name in config.loss_weights}

            for _, loss_debug_tensors in loss_dict.values():
                if loss_debug_tensors is not None:
                    rendered_results = {**rendered_results, **loss_debug_tensors}

        for k in rendered_results:
            if rendered_results[k].shape[0] in (chunk_size, last_chunk_size): # check if it can be concatted at all
                if render_keys is None or k in render_keys:
                    output_batches[k].append(rendered_results[k])

        if pbar is not None:
            pbar.step(size=len(rays_batch.cam_idxs))


    cat_output_patches = dict()

    for k, v in output_batches.items():
        if len(v) > 0:
            try:
                cat_output_patches[k] = torch.cat(v, dim=0).view(final_shape[:v[0].ndim + 1])
            except RuntimeError as e:
                pass # ignore non-reshapable outputs

    return cat_output_patches

class ScheduledOptimizer:
    def __init__(self, parameters, first_iteration: int, check_grads: bool, clip_grads: bool, scheduler_factory=None, **optimizer_kwargs) -> None:
        self.first_iteration = first_iteration
        self.check_grads = check_grads
        self.clip_grads = clip_grads
        self.parameters = parameters

        self.optimizer = None
        self.scheduler = None

        # try to create optimizer
        try:
            self.optimizer = torch.optim.Adam(self.parameters, **optimizer_kwargs)
            if scheduler_factory is not None:
                self.scheduler = scheduler_factory(self.optimizer)
        except ValueError as e:
            if "empty parameter list" in str(e):
                print("No parameters, optimizer disabled.")
            else:
                raise
        
    
    def zero_grad(self, it: int):
        if self.optimizer is not None and it >= self.first_iteration:
            self.optimizer.zero_grad(set_to_none=True)

    def step(self, it: int, loss: float):
        if self.optimizer is not None and it >= self.first_iteration:

            # check grads
            
            if self.check_grads:
                for param in self.parameters:
                    if param.grad is None:
                        print("Warning: Grad of parameter tensor", param.shape, param.dtype, "does not exist, skipping...")
                    elif param.grad.isnan().any():
                        param.grad = param.grad.nan_to_num(0.0)
                        print("Warning: Grad of parameter tensor", param.shape, param.dtype, "is nan. Setting to 0...")

            # clip grads

            if self.clip_grads:
                torch.nn.utils.clip_grad_value_(self.parameters, 1.0)

            # apply grads

            self.optimizer.step()
            
            # run scheduler

            if self.scheduler is not None:
                if isinstance(self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(metrics=loss)
                else:
                    self.scheduler.step()

    def load(self, path: Path):
        self.optimizer.load_state_dict(torch.load(path).state_dict())
        self.optimizer.zero_grad()

    def save(self, path: Path):
        if self.optimizer is not None:
            torch.save(self.optimizer, path)
        else:
            warnings.warn(
                f"Trying to save an unused optimizer to '{path}', "
                f"this will be ignored."
            )