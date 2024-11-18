# ------------------------------------------------------------------------------------
# Modified from NeRF-Factory (https://github.com/kakaobrain/nerf-factory)
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF (https://github.com/bmild/nerf)
# Copyright (c) 2020 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------


import numpy as np
import torch
from src.model.tramnerf.mirrors.material_utils import sample_brdf_ggx_vndf

class Mirror:
    def __init__(self, mirror_points, mirror_origin, mirror_normal) -> None:
        self.points = mirror_points
        self.origin = mirror_origin
        self.normal = mirror_normal
        self.normal = self.normal / torch.linalg.norm(self.normal, dim=-1, keepdim=True)

def get_mirror_list(mirror_points, mirror_origin, mirror_normal):
    mirror_list = []
    for idx, value in enumerate(mirror_points):
        mirror_list.append(Mirror(mirror_points[idx], mirror_origin[idx], mirror_normal[idx]))

    return mirror_list

def moeller_trumbore(mirror_points, origins, viewdirs, bsz):

    epsilon = 1e-8

    if mirror_points.size()[0] == 4:
        
        v0v1 = mirror_points[2] - mirror_points[1]

        v0v1 = torch.broadcast_to(v0v1, (bsz, 3))

        v0v2 = mirror_points[0] - mirror_points[1]
        v0v2 = torch.broadcast_to(v0v2, (bsz, 3))
        pvec = torch.cross(viewdirs, v0v2, dim=1)
        det = batched_dot(v0v1, pvec)

        culling = det > epsilon
        parallel = abs(det) > epsilon

        invDet = 1 / det
        tvec = origins - torch.broadcast_to(mirror_points[1], (bsz, 3))
        u = batched_dot(tvec, pvec) * invDet

        cond1 = torch.logical_and(u > 0, u < 1)


        qvec = torch.cross(tvec, v0v1, dim=-1)
        v = batched_dot(viewdirs, qvec) * invDet
        cond2 = torch.logical_and(v > 0, u + v < 1)

        left_triangle = torch.logical_and(torch.logical_and(culling, parallel), torch.logical_and(cond1, cond2))

        v0v1 = mirror_points[0] - mirror_points[3]

        v0v1 = torch.broadcast_to(v0v1, (bsz, 3))

        v0v2 = mirror_points[2] - mirror_points[3]
        v0v2 = torch.broadcast_to(v0v2, (bsz, 3))
        pvec = torch.cross(viewdirs, v0v2, dim=1)
        det = batched_dot(v0v1, pvec)

        culling = det > epsilon
        parallel = abs(det) > epsilon

        invDet = 1 / det
        tvec = origins - torch.broadcast_to(mirror_points[3], (bsz, 3))
        u = batched_dot(tvec, pvec) * invDet

        cond1 = torch.logical_and(u > 0, u < 1)


        qvec = torch.cross(tvec, v0v1, dim=-1)
        v = batched_dot(viewdirs, qvec) * invDet
        cond2 = torch.logical_and(v > 0, u + v < 1)

        right_triangle = torch.logical_and(torch.logical_and(culling, parallel), torch.logical_and(cond1, cond2))

        return torch.logical_or(right_triangle, left_triangle)

def calc_intersection(normal, mirror_origin, origins ,direction):
    D = - batched_dot(normal, mirror_origin)
    t = - (batched_dot(normal, origins) + D) / batched_dot(normal, direction)
    return origins + t.view(origins.size()[0], 1) * direction, t

def batched_dot(a, b):
  #For the edge case of only 1 ray hitting the mirror, we don't have to use squeeze
  if a.size()[0] == 1:
    return(a[..., None, :] @ b[..., :, None])
  return (a[..., None, :] @ b[..., :, None]).squeeze()

def reflect(x, n, o):
  return (x - o) - 2 * batched_dot((x - o), n)[:, None] * n + o


def img2mse(x, y):
    return torch.mean((x - y) ** 2)


def mse2psnr(x):
    return -10.0 * torch.log(x) / np.log(10.0)


def cast_rays(t_vals, origins, directions):
    points = origins[..., None, :] + t_vals[..., None] * directions[..., None, :]
    return points


def pos_enc(x, min_deg, max_deg):
    scales = torch.tensor([2**i for i in range(min_deg, max_deg)]).type_as(x)
    xb = torch.reshape((x[..., None, :] * scales[:, None]), list(x.shape[:-1]) + [-1])
    four_feat = torch.sin(torch.cat([xb, xb + 0.5 * np.pi], dim=-1))
    return torch.cat([x] + [four_feat], dim=-1)


def volumetric_rendering(rgb, density, t_vals, dirs, white_bkgd):

    eps = 1e-10

    dists = torch.cat(
        [
            t_vals[..., 1:] - t_vals[..., :-1],
            torch.ones(t_vals[..., :1].shape, device=t_vals.device) * 1e10,
        ],
        dim=-1,
    )
    dists = dists * torch.norm(dirs[..., None, :], dim=-1)
    dists = torch.abs(dists)
    alpha = 1.0 - torch.exp(-density[..., 0] * dists)
    accum_prod = torch.cat(
        [
            torch.ones_like(alpha[..., :1]),
            torch.cumprod(1.0 - alpha[..., :-1] + eps, dim=-1),
        ],
        dim=-1,
    )

    weights = alpha * accum_prod

    comp_rgb = (weights[..., None] * rgb).sum(dim=-2)
    acc = weights.sum(dim=-1)

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc[..., None])

    return comp_rgb, acc, weights, accum_prod[:, -1]

def monte_carlo_volumetric_rendering(
        main_rgb, main_density, main_t_vals, mc_rgb, mc_density,
        mc_t_vals, mc_brdf_weights, reflected_rays,
        white_bkgd
):
    # main_
    # rgb.shape     = (*B, N, 3)
    # density.shape = (*B, N, 1)
    # t_vals.shape  = (*B, N)

    # mc_
    # rgb.shape             = (*R, M, N, 3)
    # density.shape         = (*R, M, N, 1)
    # t_vals.shape          = (*R, M, N)
    # brdf_weights.shape    = (*R, M, N)

    eps = 1e-10

    # compute along main ray

    main_dists = torch.cat(
        [
            main_t_vals[..., 1:] - main_t_vals[..., :-1],
            torch.ones(main_t_vals[..., :1].shape, device=main_t_vals.device) * 1e10,
        ],
        dim=-1,
    )  # -> (*B, N)

    main_alpha = 1.0 - torch.exp(-main_density[..., 0] * main_dists)  # -> (*B, N)
    main_accum_prod = torch.cat(
        [
            torch.ones_like(main_alpha[..., :1]),
            torch.cumprod(1.0 - main_alpha[..., :-1] + eps, dim=-1),
        ],
        dim=-1,
    )  # -> (*B, N)

    main_weights = main_alpha * main_accum_prod # Ti * (1-exp(...)) -> (*B, N)
    acc = main_weights.sum(dim=-1) # -> (*B,)

    main_comp_rgb = (main_weights[..., None] * main_rgb)  # -> (*B, N, 3)
    main_comp = main_comp_rgb.sum(dim=-2) # -> (*B, 3)

    # use accurate distances along monte carlo rays

    mc_dists_1 = torch.cat(
        [
            mc_t_vals[..., 1:] - main_t_vals[reflected_rays][..., None, :-1],
            torch.ones(mc_t_vals[..., :1].shape, device=mc_t_vals.device) * 1e10,
        ],
        dim=-1,
    )

    mc_dists_2 = torch.cat(
        [
            mc_t_vals[..., 2:] - main_t_vals[reflected_rays][..., None, :-2],
            torch.ones(mc_t_vals[..., :2].shape, device=mc_t_vals.device) * 1e10,
        ],
        dim=-1,
    )

    # we need to check which T_i the mc sample belongs to (either T_i or T_{i-1})
    mc_dists = torch.where(mc_dists_1 > 0, mc_dists_1, mc_dists_2)
    mc_alpha = 1.0 - torch.exp(-mc_density[..., 0] * mc_dists)  # -> (*R, M, N)
        
    # compute weighted sum of mc samples at each original ray sampling point

    mc_comp_rgb = mc_brdf_weights[..., None] * mc_alpha[..., None] * mc_rgb  # -> (*R, M, N, 3)
    mc_comp_mean = mc_comp_rgb.sum(dim=-3)  # -> (*R, N, 3)

    # accumulate transmittance using mean observed density

    mc_accum_prod = torch.cat(
        [
            torch.ones_like(main_alpha[reflected_rays][..., :1]),
            torch.cumprod(1.0 - mc_alpha.mean(dim=-2)[..., :-1] + eps, dim=-1)  # mean fix
        ],
        dim=-1,
    )  # -> (*B, N)

    # integrate along ray

    mc_comp = (mc_accum_prod[..., None] * mc_comp_mean).sum(dim=-2)  # -> (*R, 3)

    comp_rgb = main_comp
    comp_rgb[reflected_rays] = mc_comp  # -> (*B, 3)

    if white_bkgd:
        comp_rgb = comp_rgb + (1.0 - acc[..., None])

    return comp_rgb, acc, main_weights

def sorted_piecewise_constant_pdf(
    bins, weights, num_samples, randomized, float_min_eps=2**-32
):

    eps = 1e-5
    weight_sum = weights.sum(dim=-1, keepdims=True)
    padding = torch.fmax(torch.zeros_like(weight_sum), eps - weight_sum)
    weights = weights + padding / weights.shape[-1]
    weight_sum = weight_sum + padding

    pdf = weights / weight_sum
    cdf = torch.fmin(
        torch.ones_like(pdf[..., :-1]), torch.cumsum(pdf[..., :-1], dim=-1)
    )
    cdf = torch.cat(
        [
            torch.zeros(list(cdf.shape[:-1]) + [1], device=weights.device),
            cdf,
            torch.ones(list(cdf.shape[:-1]) + [1], device=weights.device),
        ],
        dim=-1,
    )

    s = 1 / num_samples
    if randomized:
        u = torch.rand(list(cdf.shape[:-1]) + [num_samples], device=cdf.device)
    else:
        u = torch.linspace(0.0, 1.0 - float_min_eps, num_samples, device=cdf.device)
        u = torch.broadcast_to(u, list(cdf.shape[:-1]) + [num_samples])

    mask = u[..., None, :] >= cdf[..., :, None]

    bin0 = (mask * bins[..., None] + ~mask * bins[..., :1, None]).max(dim=-2)[0]
    bin1 = (~mask * bins[..., None] + mask * bins[..., -1:, None]).min(dim=-2)[0]
    cdf0 = (mask * cdf[..., None] + ~mask * cdf[..., :1, None]).max(dim=-2)[0]
    cdf1 = (~mask * cdf[..., None] + mask * cdf[..., -1:, None]).min(dim=-2)[0]

    t = torch.clip(torch.nan_to_num((u - cdf0) / (cdf1 - cdf0), 0), 0, 1)
    samples = bin0 + t * (bin1 - bin0)

    return samples

def F_Schlick(F0 : torch.Tensor, cosTheta : torch.Tensor) -> torch.Tensor:
    d = 1 - cosTheta
    return F0 + (1 - F0) * (d**5)

def Lambda_SmithGGX(NdotV : torch.Tensor, roughness : torch.Tensor) -> torch.Tensor:
    return 0.5*(torch.sqrt(1 + roughness**2 * (1-NdotV**2) / NdotV**2) - 1)


def G1_SmithGGX(NdotV : torch.Tensor, roughness : torch.Tensor, eps=1e-8) -> torch.Tensor:
    LambdaV = Lambda_SmithGGX(NdotV, roughness)
    return 1 / (1 + LambdaV)


def G2_SmithGGX(NdotL : torch.Tensor, NdotV : torch.Tensor, roughness : torch.Tensor, eps=1e-8) -> torch.Tensor:
    LambdaV = Lambda_SmithGGX(NdotV, roughness)
    LambdaL = Lambda_SmithGGX(NdotL, roughness)
    return 1 / (1 + LambdaV + LambdaL)

def bdot(a, b):
    # batched dot product over last dimension
    return (a[..., None, :] @ b[..., None])[..., 0, 0]

def reflect(v, n, o=None):
    # reflects vector v at plane through o with normal n
    v = v if o is None else v - o
    v = v - 2 * bdot(v, n)[..., None] * n
    return v if o is None else v + o

def rotate(v, axis, cos_angle, o=None):
    # rotates vector v around axis with given cos(angle), o being center of rotation
    v = v if o is None else v - o
    sin_angle = torch.sqrt(1 - cos_angle ** 2)
    v =  v * cos_angle + torch.cross(axis, v, dim=-1) * sin_angle + axis * bdot(axis, v)[..., None] * (1 - cos_angle)
    return v if o is None else v + o

def get_axis_cos_angle(a, b):
    # get rotation axis and cos(angle) to rotate a onto b
    axis = torch.cross(a, b, dim=-1)
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdims=True)
    cos_angle = bdot(a, b) / (torch.linalg.norm(a, dim=-1, keepdims=True) * torch.linalg.norm(b, dim=-1, keepdims=True))
    return axis, cos_angle

def rotate_a_to_b(v, a, b):
    # rotates v such that a is rotated onto b
    if torch.isclose(a, b).all():
        return v
    else:
        return rotate(v, *get_axis_cos_angle(a, b))

def ggx_like(
    monte_carlo_samples: torch.Tensor,
    discretized_ggx_cdf: torch.Tensor,
    mirror: Mirror,
    viewdirs: torch.Tensor,
    reflected_viewdirs: torch.Tensor,
    t_offset: torch.Tensor,
    intersections: torch.Tensor,
    variance: float,
    monte_carlo_rays: int,
    use_same_rays: bool
):
    bsz, num_samples = monte_carlo_samples.size()[:2]
    num_main_samples = num_samples // monte_carlo_rays

    if use_same_rays:
        # use same viewdir for every sample
        viewdirs = torch.tile(viewdirs[:, None], (1, monte_carlo_rays, 1))
    else:
        # use individual viewdir for every sample
        viewdirs = torch.tile(viewdirs[:, None], (1, num_samples, 1)) 

    if variance < 0:
        raise NotImplementedError("TODO: refactor the case: variance = 0")
    else:
        # rotate viewdirs such that mirror normal points to z+
        z = torch.tensor([0.0, 0.0, 1.0], device=monte_carlo_samples.device)        
        viewdirs_local = rotate_a_to_b(viewdirs, mirror.normal[None, None], z[None, None])

        rand_uv = torch.rand(*viewdirs_local.shape[:-1], 2, device=viewdirs_local.device)
        random_d_local, brdf_weights = sample_brdf_ggx_vndf(
            uv=rand_uv,
            view_dir=-viewdirs_local,
            roughness=torch.tensor([variance], device=viewdirs_local.device)
        )
        random_d = rotate_a_to_b(random_d_local, z[None, None], mirror.normal[None, None])

        if use_same_rays:
            # repeat the sampled half_vector for all samples along the ray
            viewdirs = torch.repeat_interleave(viewdirs, repeats=num_main_samples, dim=1)
            random_d = torch.repeat_interleave(random_d, repeats=num_main_samples, dim=1)
            brdf_weights = torch.repeat_interleave(brdf_weights, repeats=num_main_samples, dim=1)
    
    # get sample t values
    mids = 0.5 * (t_offset[:, 1:, 0] + t_offset[:, :-1, 0])
    upper = torch.cat([mids, t_offset[:, -1:, 0]], -1)
    lower = torch.cat([t_offset[:, :1, 0], mids], -1)
    t_rand = torch.rand((bsz, num_samples), device=monte_carlo_samples.device)
    random_t = lower + (upper - lower) * t_rand

    # compute new sample locations (parameterized and 3D)
    # divide by rv_dot_l to take sample from same interval as
    # original stratified sampling has used (->| instead of ->))
    rv_dot_l = bdot(reflected_viewdirs, random_d)[:, :, None]
    sample_results = intersections[:, None, :] + random_d * random_t[:, :, None] / rv_dot_l
    random_t_norm = random_t / torch.linalg.norm(random_d, dim=-1)
    
    # compute dot products for BRDF weighting
    return random_t_norm, sample_results, random_d, brdf_weights


def randn_like(
        monte_carlo_coords: torch.Tensor, t_offset: torch.Tensor,
        final_viewdirs, true_reflections, reflected_rays,
        variance: float, monte_carlo_rays: int
):
    monte_carlo_reflections = true_reflections[reflected_rays]
    monte_carlo_viewdirs = final_viewdirs[reflected_rays]

    monte_offset = torch.randn_like(monte_carlo_coords) * variance 
    monte_offset *= t_offset
    monte_carlo_reflections = torch.tile(monte_carlo_reflections, (1,monte_carlo_rays))
    monte_carlo_coords = monte_carlo_coords + monte_offset
    monte_carlo_viewdirs = torch.tile(monte_carlo_viewdirs, (1,monte_carlo_rays,1))
    return monte_carlo_coords, monte_carlo_viewdirs


def sample_along_rays(
    rays_o,
    rays_d,
    num_samples,
    near,
    far,
    randomized,
    lindisp,
    num_bounces, 
    mirrors,
    monte_carlo_rays,
    variance,
    discretized_ggx_cdf,
    use_same_rays
):
    bsz = rays_o.shape[0]
    t_vals = torch.linspace(0.0, 1.0, num_samples + 1, device=rays_o.device)
    if lindisp:
        t_vals = 1.0 / (1.0 / near * (1.0 - t_vals) + 1.0 / far * t_vals)
    else:
        t_vals = near * (1.0 - t_vals) + far * t_vals

    if randomized:
        mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
        upper = torch.cat([mids, t_vals[..., -1:]], -1)
        lower = torch.cat([t_vals[..., :1], mids], -1)
        t_rand = torch.rand((bsz, num_samples + 1), device=rays_o.device)
        t_vals = lower + (upper - lower) * t_rand
    else:
        t_vals = torch.broadcast_to(t_vals, (bsz, num_samples + 1))

    coords = cast_rays(t_vals, rays_o, rays_d)

    initial_intersections = torch.ones_like(rays_d, dtype=torch.float32)
    final_viewdirs = torch.tile(rays_d[:, None, :], (1, num_samples + 1, 1))
    for mirror in mirrors:
        #check which rays hit the inside of the mirror
        reflected_rays = moeller_trumbore(mirror.points, rays_o, rays_d, bsz)

        mirror_origin = torch.broadcast_to(mirror.origin, (bsz, num_samples + 1, 3))
        mirror_normal = torch.broadcast_to(mirror.normal, (bsz, num_samples + 1, 3))

        #points with a dot porduct below zero are behind the mirror and need to be reflected
        normal_sample_dot = batched_dot(coords - mirror_origin, mirror_normal)
        mirror_mask = normal_sample_dot < 0
        intersections, t = calc_intersection(torch.broadcast_to(mirror.normal, (bsz,3)),torch.broadcast_to(mirror.origin, (bsz, 3)) ,rays_o, rays_d)
        t_bool = t > 0
        true_reflections = reflected_rays & t_bool
        true_reflections = torch.broadcast_to(true_reflections.view((bsz,1)), (bsz, num_samples + 1)) & mirror_mask
        coords[true_reflections] = reflect(coords[true_reflections], mirror_normal[true_reflections], mirror_origin[true_reflections])
        initial_intersections[reflected_rays] = intersections[reflected_rays]

    viewdirs = reflect(rays_d, mirrors[0].normal)
    temp_viewdirs =  torch.tile(viewdirs[:, None, :], (1, num_samples + 1, 1))

    final_viewdirs[true_reflections] = temp_viewdirs[true_reflections]
    
    monte_carlo_coords = cast_rays(t_vals, rays_o, rays_d)[reflected_rays]
    monte_carlo_reflections = true_reflections[reflected_rays]

    monte_carlo_coords = torch.tile(monte_carlo_coords, (1, monte_carlo_rays, 1))
    
    t_offset = torch.tile((t_vals[reflected_rays] - t[reflected_rays][:,None])[:,:,None], (1, monte_carlo_rays, 1))
    monte_carlo_reflections = torch.tile(monte_carlo_reflections, (1,monte_carlo_rays))

    if monte_carlo_coords.shape[0] > 0:
        monte_carlo_t, monte_carlo_coords_changed, monte_carlo_viewdirs, brdf_weights = ggx_like(
            monte_carlo_coords,
            discretized_ggx_cdf,
            mirrors[0],
            rays_d[reflected_rays],
            torch.tile(final_viewdirs[reflected_rays], (1, monte_carlo_rays, 1)),
            t_offset,
            intersections[reflected_rays],
            variance,
            monte_carlo_rays,
            use_same_rays
        )
        monte_carlo_coords[monte_carlo_reflections] = monte_carlo_coords_changed[monte_carlo_reflections]
        return t_vals, coords, final_viewdirs, monte_carlo_t, monte_carlo_coords, monte_carlo_viewdirs, reflected_rays, true_reflections, t, brdf_weights
    else:
        return t_vals, coords, final_viewdirs, None, monte_carlo_coords, None, reflected_rays, true_reflections, t, None


def sample_pdf(bins, weights, origins, directions, t_vals, num_samples, randomized, num_bounces, mirrors, monte_carlo_rays, variance, discretized_ggx_cdf, use_same_rays):


    t_samples = sorted_piecewise_constant_pdf(
        bins, weights, num_samples, randomized
    ).detach()
    t_vals = torch.sort(torch.cat([t_vals, t_samples], dim=-1), dim=-1).values
    coords = cast_rays(t_vals, origins, directions)

    bsz = coords.size()[0]
    num_samples = coords.size()[1]
    rays_d = directions
    rays_o = origins

    initial_intersections = torch.ones_like(rays_d, dtype=torch.float32)
    final_viewdirs = torch.tile(rays_d[:, None, :], (1, num_samples, 1))
    for mirror in mirrors:


        reflected_rays = moeller_trumbore(mirror.points, rays_o, rays_d, bsz)

        mirror_origin = torch.broadcast_to(mirror.origin, (bsz, num_samples, 3))
        mirror_normal = torch.broadcast_to(mirror.normal, (bsz, num_samples, 3))

        normal_sample_dot = batched_dot(coords - mirror_origin, mirror_normal)
        mirror_mask = normal_sample_dot < 0
        intersections, t = calc_intersection(torch.broadcast_to(mirror.normal, (bsz,3)),torch.broadcast_to(mirror.origin, (bsz, 3)) ,rays_o, rays_d)
        t_bool = t > 0
        true_reflections = reflected_rays & t_bool
        true_reflections = torch.broadcast_to(true_reflections.view((bsz,1)), (bsz, num_samples)) & mirror_mask
        coords[true_reflections] = reflect(coords[true_reflections], mirror_normal[true_reflections], mirror_origin[true_reflections])
        initial_intersections[reflected_rays] = intersections[reflected_rays]

    viewdirs = coords[:,-1] - initial_intersections
    viewdirs = viewdirs / torch.linalg.norm(viewdirs, dim=-1, keepdim=True)
    temp_viewdirs =  torch.tile(viewdirs[:, None, :], (1, num_samples, 1))


    final_viewdirs[true_reflections] = temp_viewdirs[true_reflections]

    monte_carlo_coords = cast_rays(t_vals, rays_o, rays_d)[reflected_rays]
    monte_carlo_reflections = true_reflections[reflected_rays]

    monte_carlo_coords = torch.tile(monte_carlo_coords, (1, monte_carlo_rays, 1))

    t_offset = torch.tile((t_vals[reflected_rays] - t[reflected_rays][:,None])[:,:,None], (1, monte_carlo_rays, 1))
    monte_carlo_reflections = torch.tile(monte_carlo_reflections, (1,monte_carlo_rays))


    if monte_carlo_coords.shape[0] > 0:
        monte_carlo_t, monte_carlo_coords_changed, monte_carlo_viewdirs, brdf_weights = ggx_like(
            monte_carlo_coords, discretized_ggx_cdf, mirrors[0], rays_d[reflected_rays],
            torch.tile(final_viewdirs[reflected_rays], (1, monte_carlo_rays, 1)),
            t_offset, intersections[reflected_rays], variance, monte_carlo_rays,
            use_same_rays
        )
        monte_carlo_coords[monte_carlo_reflections] = monte_carlo_coords_changed[monte_carlo_reflections]
        return t_vals, coords, final_viewdirs, monte_carlo_t, monte_carlo_coords, monte_carlo_viewdirs, reflected_rays, true_reflections, t, brdf_weights
    else:
        return t_vals, coords, final_viewdirs, None, monte_carlo_coords, None, reflected_rays, true_reflections, t, None
