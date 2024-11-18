import warnings
import torch
import torch.utils.hooks
from typing import NamedTuple

from .materials import Material
from src.data.utils import Rays, SampledRays
from .utils import masked_less
from ..utils import batched_dot
from tensor_shape_assert import check_tensor_shapes, ShapedTensor, get_shape_variables


class Intersection(NamedTuple):
    t: torch.Tensor # (n,)
    world: torch.Tensor # (n, 3)
    normal: torch.Tensor # (n, 3)
    is_valid: torch.Tensor # (n,)

    def get_closest(
            self,
            other: 'Intersection',
            min_t: torch.Tensor = None # (n,)
    ):
        self_valid = self.is_valid
        other_valid = other.is_valid
        
        if min_t is not None:
            self_valid = self_valid & (self.t > min_t)
            other_valid = other_valid & (other.t > min_t)

        self_is_closer = masked_less(self.t, other.t, self_valid, other_valid)
        self_is_closer_3d = self_is_closer[:, None].expand(-1, 3)
        return Intersection(
            t=torch.where(self_is_closer, self.t, other.t),
            world=torch.where(self_is_closer_3d, self.world, other.world),
            normal=torch.where(self_is_closer_3d, self.normal, other.normal),
            is_valid=self.is_valid | other.is_valid
        )
    
    def detach(self):
        return Intersection(
            t=self.t.detach(),
            world=self.world.detach(),
            normal=self.normal.detach(),
            is_valid=self.is_valid.detach(),
        )
    
class Mirror:
    def transformed(self, center: torch.Tensor, scale: float) -> 'Mirror':
        raise NotImplementedError

    def get_intersections(
            self,
            rays: Rays,
            t_mids: ShapedTensor["n r"] = None,
    ) -> tuple[Intersection, dict[str, torch.Tensor]]:
        raise NotImplementedError

    
def get_is_ray_reflected(intersections):
    is_intersection_in_front_of_camera = intersections.t > 0 # (n_rays,)
    is_ray_reflected = intersections.is_valid & is_intersection_in_front_of_camera # (n_rays,)
    return is_ray_reflected


def get_is_coord_reflected(is_ray_reflected, num_rays, num_samples, t_mean, coords, mirror_origin, mirror_normal, last_intersections=None):
    is_coord_reflected = is_ray_reflected[:, None].expand(num_rays, num_samples)
    
    if last_intersections is not None:
        is_coord_reflected = is_coord_reflected & (t_mean > last_intersections.t[:, None])

    coord_dot_normal = batched_dot(coords - mirror_origin, mirror_normal) # (n_rays, n_samples)
    is_coord_behind_mirror = coord_dot_normal < 0 # (n_rays, n_samples)
    is_coord_reflected = is_coord_reflected & is_coord_behind_mirror

    return is_coord_reflected


@check_tensor_shapes(experimental_enable_autogen_constraints=True)
def reflect_rays(
        rays: Rays, # (batch_size, 3)
        t_vals: ShapedTensor["batch_size, rays_per_pixel, samples_per_ray+1"],
        t_mean: ShapedTensor["batch_size, rays_per_pixel, samples_per_ray"],
        mirrors: list[Mirror],
        materials: list[Material],
        num_bounces: int,
        num_ray_samples: int,
        return_debug_images: bool = True,
        dry_run: bool = False,
        use_dense_sampling: bool = False
) -> tuple[
    ShapedTensor["batch_size, rays_per_pixel, samples_per_ray, 3"], # coords
    ShapedTensor["batch_size, rays_per_pixel, samples_per_ray, 3"], # final_viewdirs
    ShapedTensor["batch_size, rays_per_pixel, samples_per_ray"],    # sample_weights
    list[Intersection],                                             # all_intersections
    ShapedTensor["num_bounces, batch_size, rays_per_pixel, 3"],     # all_reflected_viewdirs
    ShapedTensor["num_bounces, batch_size, rays_per_pixel"],        # all_ray_weights
    dict                                                            # debug_tensors
]:
    # get material
    material = materials[0]

    # get some shape info
    num_batch, num_ray_samples, num_samples = get_shape_variables("batch_size rays_per_pixel samples_per_ray")
    num_rays = num_batch * num_ray_samples
    device = rays.origins.device

    # compute 3D coordinates
    coords = rays.origins[:, None, None, :] + rays.viewdirs[:, None, None, :] * t_mean[:, :, :, None] # (b, r, s, 3)
    coords = coords.view(-1, num_samples, 3) # (br, s, 3)
    t_mean = t_mean.view(-1, num_samples)

    # copy input data
    new_origins = rays.origins[:, None, :].tile(1, num_ray_samples, 1).view(-1, 3)    # (br, 3)
    new_viewdirs = rays.viewdirs[:, None, :].tile(1, num_ray_samples, 1).view(-1, 3)  # (br, 3)
    
    # allocate tensors
    sample_weights = torch.ones_like(coords[:, :, 0]) # (br, s)
    ray_num_bounces = torch.zeros(num_rays, dtype=torch.int32, device=device)

    # allocate debug tensors
    if return_debug_images:
        first_reflection_direction = torch.zeros_like(new_viewdirs) # (br, 3)
    
    final_viewdirs = torch.tile(new_viewdirs[:, None, :], (1, num_samples, 1)) # (br, s, 3)

    # allocate intersection info
    all_intersections: list[Intersection] = []
    all_reflected_viewdirs = torch.zeros((num_bounces, num_rays, 3), device=device)
    all_ray_weights = torch.zeros((num_bounces, num_rays), device=device) * torch.nan

    debug_tensors = dict()

    for i in range(num_bounces):
        # find closest intersection
        intersections = Intersection(
            t=torch.empty((num_rays,), device=device),
            world=torch.empty((num_rays, 3), device=device),
            normal=torch.empty((num_rays, 3), device=device),
            is_valid=torch.zeros(num_rays, dtype=torch.bool, device=device)
        )
        
        for j, mirror in enumerate(mirrors):
            sampled_rays = SampledRays(
                origins=new_origins,
                viewdirs=new_viewdirs,
                radii=rays.radii.repeat_interleave(repeats=num_ray_samples, dim=0),
                cam_idxs=None if rays.cam_idxs is None else rays.cam_idxs.repeat_interleave(repeats=num_ray_samples, dim=0),
                xy=None if rays.xy is None else rays.xy.repeat_interleave(repeats=num_ray_samples, dim=0)
            )

            cur_intersections, mirror_debug_tensors = mirror.get_intersections(
                rays=sampled_rays,
                t_mids=t_mean,
            )

            if i > 0:
                # This fixes the issue that currently all get_intersection implementations
                # are using rays.origins (=last intersection point) and rays.viewdirs
                # (=reflected direction) to determine the rays and therefore
                # have no idea of the actual t position on the piecewise ray.
                cur_intersections = Intersection(
                    t=cur_intersections.t + all_intersections[i-1].t,
                    world=cur_intersections.world,
                    normal=cur_intersections.normal,
                    is_valid=cur_intersections.is_valid
                )

            # fix for double-sided triangles: intersection needs a minimum dist to (new) ray origin
            somewhat_distant_to_origin = (new_origins - cur_intersections.world).norm(dim=-1) > 1e-2
            cur_intersections = Intersection(
                t=cur_intersections.t,
                world=cur_intersections.world,
                normal=cur_intersections.normal,
                is_valid=cur_intersections.is_valid & somewhat_distant_to_origin
            )

            intersections = intersections.get_closest(
                other=cur_intersections,
                min_t=0 if i == 0 else all_intersections[i-1].t + 1e-2
            )

            debug_tensors = {
                **debug_tensors,
                **{f"mirror_{j}_bounce_{i}_{k}": v for k, v in mirror_debug_tensors.items()}
            }

            # add intersection test results for each bounce and mirror
            debug_tensors = {
                **debug_tensors,
                f"mirror_{j}_bounce_{i}_t": cur_intersections.t.view(num_batch, num_ray_samples)[:, 0],
                f"mirror_{j}_bounce_{i}_normal": cur_intersections.normal.view(num_batch, num_ray_samples, 3)[:, 0],
                f"mirror_{j}_bounce_{i}_valid": cur_intersections.is_valid.view(num_batch, num_ray_samples)[:, 0],
            }

        # expand origins and normals along sample axis
        mirror_origin = intersections.world[:, None].expand(num_rays, num_samples, 3)
        mirror_normal = intersections.normal[:, None].expand(num_rays, num_samples, 3)
        
        # check if intersection is valid
        is_ray_reflected = get_is_ray_reflected(intersections)

        # check which coords need to be reflected
        is_coord_reflected = get_is_coord_reflected(
            is_ray_reflected=is_ray_reflected,
            num_rays=num_rays,
            num_samples=num_samples,
            t_mean=t_mean,
            coords=coords,
            mirror_origin=mirror_origin,
            mirror_normal=mirror_normal,
            last_intersections=all_intersections[i-1] if i > 0 else None
        )

        ray_new_viewdirs = new_viewdirs.view(num_batch, num_ray_samples, 3)
        ray_mirror_normals = mirror_normal[:, 0].view(num_batch, num_ray_samples, 3)
     
        ray_new_viewdirs = ray_new_viewdirs.view(-1, 1, 3)
        ray_mirror_normals = ray_mirror_normals.view(-1, 1, 3)
        
        reflected_viewdirs, ray_weights = material.sample_reflected_viewdirs(
            viewdirs=ray_new_viewdirs,
            normals=ray_mirror_normals,
        ) # (n o 3), (n o)

        reflected_viewdirs = reflected_viewdirs.view(num_rays, 3)
        ray_weights = ray_weights.view(num_rays) * (1 / num_ray_samples)

        # apply reflection
        # TODO optimize this by using scatter?
        if not dry_run:
            new_viewdirs = torch.where(is_ray_reflected[:, None], reflected_viewdirs, new_viewdirs)
            final_viewdirs = torch.where(is_coord_reflected[:, :, None], new_viewdirs[:, None, :], final_viewdirs)
            new_origins = torch.where(is_ray_reflected[:, None], intersections.world, new_origins)
            new_coords = new_origins[:, None] + new_viewdirs[:, None] * (t_mean[:, :, None] - intersections.t[:, None, None])
            coords = torch.where(is_coord_reflected[:, :, None], new_coords, coords)
        else:
            warnings.warn("Setting dry_run = True currently only considers a single bounce.")
        
        # memorize intersection info for resampling
        all_intersections.append(intersections)
        all_reflected_viewdirs[i] = reflected_viewdirs
        all_ray_weights[i] = ray_weights

        ray_num_bounces[is_ray_reflected] += 1

        # generate debug ouptut
        if return_debug_images:
            if i == 0:
                first_reflection_direction[is_ray_reflected] = new_viewdirs[is_ray_reflected]

        # abort if no rays were reflected (and we don't need debug images)
        if not return_debug_images and not is_ray_reflected.any():
            break

    output_shape = (num_batch, num_ray_samples)

    if return_debug_images:
        assert not final_viewdirs.isnan().any()

        # just take the first ray here for debugging purposes
        debug_tensors = {
            **debug_tensors,
            "ray_num_bounces": ray_num_bounces.view(output_shape)[:, 0],
            "first_reflection_direction": first_reflection_direction.view(*output_shape, 3)[:, 0],
            "final_viewdirs": new_viewdirs.view(*output_shape, 3)[:, 0]
        }

    assert not coords.isnan().any()

    return_tuple = (
        coords.view(*output_shape, num_samples, 3),
        final_viewdirs.view(*output_shape, num_samples, 3),
        sample_weights.view(*output_shape, num_samples),
        all_intersections,
        all_reflected_viewdirs.view(num_bounces, *output_shape, 3),
        all_ray_weights.view(num_bounces, *output_shape),
        debug_tensors
    )

    return return_tuple


@check_tensor_shapes()
def reflect_rays_known(
        rays: Rays, # (batch_size, 3)
        t_mean: ShapedTensor["batch_size, rays_per_pixel, samples_per_ray"],
        all_intersections: list[Intersection],
        all_reflected_viewdirs: ShapedTensor["num_bounces, batch_size, rays_per_pixel, 3"],
        all_ray_weights: ShapedTensor["num_bounces, batch_size, rays_per_pixel"],
        return_debug_images: bool = True,
        dry_run: bool = False
) -> tuple[
    ShapedTensor["batch_size, rays_per_pixel, samples_per_ray, 3"],
    ShapedTensor["batch_size, rays_per_pixel, samples_per_ray, 3"],
    ShapedTensor["batch_size, rays_per_pixel, samples_per_ray"],
    dict
]:
    # get some shape info
    num_batch, num_ray_samples, num_samples = get_shape_variables("batch_size rays_per_pixel samples_per_ray")
    num_rays = num_batch * num_ray_samples
    device = rays.origins.device

    # compute 3D coordinates
    coords = rays.origins[:, None, None, :] + rays.viewdirs[:, None, None, :] * t_mean[:, :, :, None] # (b, r, s, 3)
    coords = coords.view(-1, num_samples, 3) # (br, s, 3)
    t_mean = t_mean.view(-1, num_samples)

    # copy input data
    new_origins = rays.origins[:, None, :].tile(1, num_ray_samples, 1).view(-1, 3)    # (br, 3)
    new_viewdirs = rays.viewdirs[:, None, :].tile(1, num_ray_samples, 1).view(-1, 3)  # (br, 3)
    
    # allocate weights tensor
    sample_weights = torch.ones_like(coords[:, :, 0]) # (br, s)

    # allocate debug tensors
    if return_debug_images:
        ray_num_bounces = torch.zeros(num_rays, dtype=torch.int32, device=device)
        first_reflection_direction = torch.zeros_like(new_viewdirs) # (br, 3)
    
    final_viewdirs = torch.tile(new_viewdirs[:, None, :], (1, num_samples, 1)) # (br, s, 3)

    # for each bounce
    for i, intersections in enumerate(all_intersections):
        # expand origins and normals along sample axis
        mirror_origin = intersections.world[:, None].expand(num_rays, num_samples, 3)
        mirror_normal = intersections.normal[:, None].expand(num_rays, num_samples, 3)
        
        # check if intersection is valid
        is_ray_reflected = get_is_ray_reflected(intersections)

        # check which coords need to be reflected
        is_coord_reflected = get_is_coord_reflected(
            is_ray_reflected=is_ray_reflected,
            num_rays=num_rays,
            num_samples=num_samples,
            t_mean=t_mean,
            coords=coords,
            mirror_origin=mirror_origin,
            mirror_normal=mirror_normal,
            last_intersections=all_intersections[i-1] if i > 0 else None
        )

        reflected_viewdirs = all_reflected_viewdirs[i].view(num_rays, 3)
        ray_weights = all_ray_weights[i].view(num_rays)

        # apply reflection
        if not dry_run:
            new_viewdirs = torch.where(is_ray_reflected[:, None], reflected_viewdirs, new_viewdirs)
            final_viewdirs = torch.where(is_coord_reflected[:, :, None], new_viewdirs[:, None, :], final_viewdirs)
            new_origins = torch.where(is_ray_reflected[:, None], intersections.world, new_origins)
            sample_weights = torch.where(is_coord_reflected, sample_weights * ray_weights[:, None], sample_weights)
            
            new_coords = new_origins[:, None] + new_viewdirs[:, None] * (t_mean[:, :, None] - intersections.t[:, None, None])
            coords = torch.where(is_coord_reflected[:, :, None], new_coords, coords)
        else:
            warnings.warn("Setting dry_run = True currently only considers a single bounce.")

        # generate debug output
        if return_debug_images:
            if i == 0:
                first_reflection_direction[is_ray_reflected] = new_viewdirs[is_ray_reflected]
            ray_num_bounces[is_ray_reflected] += 1

        # abort if no rays were reflected
        if not is_ray_reflected.any():
            break

    output_shape = (num_rays // num_ray_samples, num_ray_samples)

    if return_debug_images:
        assert not final_viewdirs.isnan().any()

        # just take the first ray here for debugging purposes
        debug_tensors = {
            "ray_num_bounces": ray_num_bounces.view(output_shape)[:, 0],
            "first_reflection_direction": first_reflection_direction.view(*output_shape, 3)[:, 0],
            "final_viewdirs": new_viewdirs.view(*output_shape, 3)[:, 0]
        }
    else:
        debug_tensors = dict()

    return (
        coords.view(*output_shape, num_samples, 3),
        final_viewdirs.view(*output_shape, num_samples, 3),
        sample_weights.view(*output_shape, num_samples),
        debug_tensors
    )
