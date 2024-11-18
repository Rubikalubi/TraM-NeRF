# ------------------------------------------------------------------------------------
# Modified from NeRF-Factory (https://github.com/kakaobrain/nerf-factory)
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Mip-NeRF (https://github.com/google/mipnerf)
# Copyright (c) 2021 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

from typing import Any
import numpy as np
import torch
from .utils import compute_transmittance

from src.data.utils import Rays
from .mirrors.base import Mirror, Intersection, reflect_rays, reflect_rays_known
from .mirrors.analytical import TriangleMirror, CircleMirror, CylinderMirror, OpenCylinderMirror
from .mirrors.materials import Material
from .mirrors import materials
from tensor_shape_assert import ShapedTensor, check_tensor_shapes

MIRROR_TYPE_TO_CLASS = {
    "triangle": TriangleMirror,
    "circle": CircleMirror,
    "cylinder": CylinderMirror,
    "open_cylinder": OpenCylinderMirror,
}

def parse_mirror_descriptors(mirror_descriptors: list[dict], device) -> list[Mirror]:
    mirrors = []
    for descriptor in mirror_descriptors:
        other_params = dict()
        for k, v in descriptor.items():
            if k == 'type':
                continue
            try:
                other_params[k] = torch.tensor(v, device=device)
            except TypeError:
                other_params[k] = v
        mirror = MIRROR_TYPE_TO_CLASS[descriptor['type']](**other_params)
        mirrors.append(mirror)
    return mirrors

MATERIAL_TO_CLASS = {
    "opaque_mirror": materials.OpaqueMirrorMaterial,
}

def parse_material_descriptors(material_descriptors: list[dict]) -> list[Material]:
    materials = []
    for descriptor in material_descriptors:
        other_params = {k: p for k, p in descriptor.items() if k != 'type'}
        material = MATERIAL_TO_CLASS[descriptor['type']](**other_params)
        materials.append(material)
    return materials

@check_tensor_shapes()
def sample_along_rays(
    rays: Rays,
    num_samples: int,
    near: float,
    far: float,
    randomized: bool,
    lindisp: bool,
    num_bounces: int, 
    num_ray_samples: int,
    mirrors: list[Mirror],
    materials: list[Material],
    return_debug_images: bool,
) -> tuple[
    ShapedTensor["b r s+1"], # t_vals
    tuple[
        ShapedTensor["b r s 3"], # means
        ShapedTensor["b r s 3"]  # covs
    ],
    ShapedTensor["b r s 3"], # viewdirs
    ShapedTensor["b r s"],   # sample_weights
    list[Intersection],
    Any,
    # ShapedTensor["n b r 3"],   # reflected_viewdirs
    ShapedTensor["n b r"],     # ray_weights
    dict
]:
    # generate stratified samples

    bsz = rays.origins.shape[0]
    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=rays.origins.device)
    if lindisp:
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        t_vals = near * (1.0 - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand((bsz, num_ray_samples, num_samples + 1), device=rays.origins.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        t_vals = torch.broadcast_to(t_vals, (bsz, num_ray_samples, num_samples + 1))


    # reflect ray samples
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    t_means = (t0 + t1) / 2


    coords, final_viewdirs, sample_weights, intersections, reflected_viewdirs, ray_weights, debug_tensors = reflect_rays(
        rays=rays,
        t_vals=t_vals,
        t_mean=t_means,
        mirrors=mirrors,
        materials=materials,
        num_bounces=num_bounces,
        num_ray_samples=num_ray_samples,
        return_debug_images=return_debug_images,
    )

    # compute gaussians from samples

    means, covs, viewdirs = cast_rays(
        t_vals=t_vals,
        means=coords,
        viewdirs=final_viewdirs,
        radii=rays.radii
    )

    return t_vals, (means, covs), viewdirs, sample_weights, intersections, reflected_viewdirs, ray_weights, debug_tensors

@check_tensor_shapes()
def resample_along_rays(
    rays: Rays,
    intersections: list[Intersection],
    reflected_viewdirs: ShapedTensor["n b r 3"],
    ray_weights: ShapedTensor["n b r"],
    t_vals: ShapedTensor["b r s+1"],
    weights: ShapedTensor["b r s"],
    randomized: bool,
    stop_level_grad: bool,
    resample_padding: float,
    return_debug_images: bool,
) -> tuple[
    ShapedTensor["b r sp1"],
    tuple[
        ShapedTensor["b r s 3"],
        ShapedTensor["b r s 3"]
    ],
    ShapedTensor["b r s 3"],
    ShapedTensor["b r s"],
    dict
]:
    weights_pad = torch.cat([weights[..., :1], weights, weights[..., -1:]], dim=-1)
    weights_max = torch.fmax(weights_pad[..., :-1], weights_pad[..., 1:])
    weights_blur = 0.5 * (weights_max[..., :-1] + weights_max[..., 1:])

    weights = weights_blur + resample_padding

    new_t_vals = sorted_piecewise_constant_pdf(
        t_vals, weights, t_vals.shape[-1], randomized
    )
    if stop_level_grad:
        new_t_vals = new_t_vals.detach()

    # reflect ray samples

    t0 = new_t_vals[..., :-1]
    t1 = new_t_vals[..., 1:]
    new_t_means = (t0 + t1) / 2

    coords, final_viewdirs, sample_weights, debug_tensors = reflect_rays_known(
        rays=rays,
        t_mean=new_t_means,
        all_intersections=intersections,
        all_reflected_viewdirs=reflected_viewdirs,
        all_ray_weights=ray_weights,
        return_debug_images=return_debug_images,
    )

    # compute gaussians from samples

    means, covs, viewdirs = cast_rays(
        t_vals=new_t_vals,
        means=coords,
        viewdirs=final_viewdirs,
        radii=rays.radii
    )

    return new_t_vals, (means, covs), viewdirs, sample_weights, debug_tensors


# 2**(-52) is the minimum epsilon value
def sorted_piecewise_constant_pdf(
    bins, weights, num_samples, randomized, float_min_eps=2**-32
):
    eps = 1e-5
    weight_sum = weights.sum(dim=-1, keepdims=True)
    padding = torch.fmax(torch.zeros_like(weight_sum), eps - weight_sum)
    weights += padding / weights.shape[-1]
    weight_sum += padding

    pdf = weights / weight_sum
    cdf = torch.fmin(
        torch.ones_like(pdf[..., :-1]), torch.cumsum(pdf[..., :-1], axis=-1)
    )
    cdf = torch.cat(
        [
            torch.zeros(list(cdf.shape[:-1]) + [1], device=weights.device),
            cdf,
            torch.ones(list(cdf.shape[:-1]) + [1], device=weights.device),
        ],
        axis=-1,
    )

    if randomized:
        s = 1 / num_samples
        u = torch.arange(num_samples, device=weights.device) * s
        u += torch.rand_like(u) * (s - float_min_eps)
        u = torch.fmin(u, torch.ones_like(u) * (1.0 - float_min_eps))
    else:
        u = torch.linspace(0.0, 1.0 - float_min_eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    mask = u[..., None, :] >= cdf[..., :, None]

    bin0 = (mask * bins[..., None] + ~mask * bins[..., :1, None]).max(dim=-2)[0]
    bin1 = (~mask * bins[..., None] + mask * bins[..., -1:, None]).min(dim=-2)[0]
    # Debug Here
    cdf0 = (mask * cdf[..., None] + ~mask * cdf[..., :1, None]).max(dim=-2)[0]
    cdf1 = (~mask * cdf[..., None] + mask * cdf[..., -1:, None]).min(dim=-2)[0]

    t = torch.clip(torch.nan_to_num((u - cdf0) / (cdf1 - cdf0), 0), 0, 1)
    samples = bin0 + t * (bin1 - bin0)

    return samples


def integrated_pos_enc(samples, min_deg, max_deg):
    x, x_cov_diag = samples
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)]).type_as(x)
    shape = list(x.shape[:-1]) + [-1]
    y = torch.reshape(x[..., None, :] * scales[:, None], shape)
    y_var = torch.reshape(x_cov_diag[..., None, :] * scales[:, None] ** 2, shape)

    return expected_sin(
        torch.cat([y, y + 0.5 * np.pi], axis=-1), torch.cat([y_var] * 2, axis=-1)
    )[0]


@check_tensor_shapes()
def volumetric_rendering(
        rgb: ShapedTensor["b s 3"],
        density: ShapedTensor["b s"],
        t_vals: ShapedTensor["b s+1"],
        dirs: ShapedTensor["b 3"],
        sample_weights: ShapedTensor["b s"],
        white_bkgd: bool
    ) -> tuple[
        ShapedTensor["b 3"],
        ShapedTensor["b"],
        ShapedTensor["b"],
        ShapedTensor["b s"]
    ]:
    t_mids = 0.5 * (t_vals[..., :-1] + t_vals[..., 1:])
    
    trans, alpha = compute_transmittance(t_vals=t_vals, density=density, dirs=dirs)

    weights = alpha * trans * sample_weights

    comp_rgb = (weights[..., None] * rgb).sum(axis=-2)
    acc = weights.sum(axis=-1)
    distance = (weights * t_mids).sum(axis=-1) / (acc + 1e-8)
    distance = torch.clip(distance, t_vals[:, 0], t_vals[:, -1])

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc[..., None])

    return comp_rgb, distance, acc, weights


def pos_enc(x, min_deg, max_deg, append_identity):
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)]).type_as(x)
    xb = torch.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
    if append_identity:
        return torch.cat([x] + [four_feat], axis=-1)
    else:
        return four_feat


def expected_sin(x, x_var):
    y = torch.exp(-0.5 * x_var) * torch.sin(x)
    y_var = 0.5 * (1 - torch.exp(-2 * x_var) * torch.cos(2 * x)) - y**2
    y_var = torch.fmax(torch.zeros_like(y_var), y_var)
    return y, y_var


@check_tensor_shapes()
def lift_gaussian(
        means: ShapedTensor["b r s 3"],
        viewdirs: ShapedTensor["b r s 3"],
        t_var: ShapedTensor["b r s"],
        r_var: ShapedTensor["b r s"],
):
    # compute 3D coordinates
    coords = means
    final_viewdirs = viewdirs

    # get some shape info
    num_rays, num_ray_samples, num_samples, _ = coords.shape

    d = final_viewdirs.view(num_rays, num_ray_samples, num_samples, 3)
    d_mag_sq = torch.sum(d**2, dim=-1, keepdim=True)
    thresholds = torch.ones_like(d_mag_sq) * 1e-10
    d_mag_sq = torch.fmax(d_mag_sq, thresholds)

    d_outer_diag = d**2
    null_outer_diag = 1 - d_outer_diag / d_mag_sq
    t_cov_diag = t_var[..., None] * d_outer_diag
    xy_cov_diag = r_var[..., None] * null_outer_diag
    cov_diag = t_cov_diag + xy_cov_diag

    return coords, cov_diag, final_viewdirs


# According to the link below, numerically stable implementations are required.
# https://github.com/google/mipnerf/blob/84c969e0a623edd183b75693aed72a7e7c22902d/internal/mip.py#L88


@check_tensor_shapes()
def conical_frustum_to_gaussian(
    t0: ShapedTensor["b r s"],
    t1: ShapedTensor["b r s"],
    means: ShapedTensor["b r s 3"],
    viewdirs: ShapedTensor["b r s 3"],
    radii: ShapedTensor["b 1"]
):
    mu = (t0 + t1) / 2
    hw = (t1 - t0) / 2
    t_mean = mu + (2 * mu * hw**2) / (3 * mu**2 + hw**2)
    t_var = (hw**2) / 3 - (4 / 15) * (
        (hw**4 * (12 * mu**2 - hw**2)) / (3 * mu**2 + hw**2) ** 2
    )
    r_var = radii[:, None]**2 * (
        (mu**2) / 4
        + (5 / 12) * hw**2
        - 4 / 15 * (hw**4) / (3 * mu**2 + hw**2)
    )

    return lift_gaussian(
        means=means,
        viewdirs=viewdirs,
        t_var=t_var,
        r_var=r_var
    )


@check_tensor_shapes()
def cast_rays(
        t_vals: ShapedTensor["b r sp1"],
        means: ShapedTensor["b r s 3"],
        viewdirs: ShapedTensor["b r s 3"],
        radii: ShapedTensor["b 1"]
) -> tuple[
    ShapedTensor["b r s 3"],
    ShapedTensor["b r s 3"],
    ShapedTensor["b r s 3"]
]:
    t0 = t_vals[..., :-1]
    t1 = t_vals[..., 1:]
    means, covs, viewdirs = conical_frustum_to_gaussian(
        t0=t0,
        t1=t1,
        means=means,
        viewdirs=viewdirs,
        radii=radii
    )
    return means, covs, viewdirs
