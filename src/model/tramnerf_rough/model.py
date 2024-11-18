# ------------------------------------------------------------------------------------
# Modified from NeRF-Factory (https://github.com/kakaobrain/nerf-factory)
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from Mip-NeRF (https://github.com/google/mipnerf)
# Copyright (c) 2021 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------

import warnings

import gin
import torch
import torch.nn as nn
import torch.nn.init as init

import src.model.tramnerf_rough.helper as helper
from src.data.utils import Rays
from src.model.tramnerf.helper import parse_mirror_descriptors

class Mirror(nn.Module):
    def __init__(self, mirror_points, mirror_origin, mirror_normal) -> None:
        super(Mirror, self).__init__()
        self.points = torch.nn.Parameter(mirror_points)
        self.origin = torch.nn.Parameter(mirror_origin)
        self.normal = mirror_normal
        self.normal = torch.nn.Parameter(self.normal / torch.linalg.norm(self.normal, dim=-1, keepdim=True))

def get_mirror_list(mirror_points, mirror_origin, mirror_normal):
    mirror_list = []
    for idx, value in enumerate(mirror_points):
        mirror_list.append(Mirror(mirror_points[idx], mirror_origin[idx], mirror_normal[idx]))

    return torch.nn.ModuleList(mirror_list)

@gin.configurable()
class TraMNeRFRoughMLP(nn.Module):
    def __init__(
        self,
        min_deg_point,
        max_deg_point,
        deg_view,
        netdepth: int = 8,
        netwidth: int = 256,
        netdepth_condition: int = 1,
        netwidth_condition: int = 128,
        skip_layer: int = 4,
        input_ch: int = 3,
        input_ch_view: int = 3,
        num_rgb_channels: int = 3,
        num_density_channels: int = 1,
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(TraMNeRFRoughMLP, self).__init__()

        self.net_activation = nn.ReLU()
        pos_size = ((max_deg_point - min_deg_point) * 2 + 1) * input_ch
        view_pos_size = (deg_view * 2 + 1) * input_ch_view
        
        init_fn_ = init.xavier_uniform_

        init_layer = nn.Linear(pos_size, netwidth)
        init_fn_(init_layer.weight)
        pts_linear = [init_layer]

        for idx in range(netdepth - 1):
            if idx % skip_layer == 0 and idx > 0:
                module = nn.Linear(netwidth + pos_size, netwidth)
            else:
                module = nn.Linear(netwidth, netwidth)
            init_fn_(module.weight)
            pts_linear.append(module)

        self.pts_linears = nn.ModuleList(pts_linear)

        views_linear = [nn.Linear(netwidth + view_pos_size, netwidth_condition)]

        for idx in range(netdepth_condition - 1):
            layer = nn.Linear(netwidth_condition, netwidth_condition)
            init_fn_(layer.weight)
            views_linear.append(layer)

        self.views_linear = nn.ModuleList(views_linear)

        self.bottleneck_layer = nn.Linear(netwidth, netwidth)
        self.density_layer = nn.Linear(netwidth, num_density_channels)
        self.rgb_layer = nn.Linear(netwidth_condition, num_rgb_channels)

        init_fn_(self.bottleneck_layer.weight)
        init_fn_(self.density_layer.weight)
        init_fn_(self.rgb_layer.weight)

    def forward(self, x, condition):

        num_samples, feat_dim = x.shape[1:]
        x = x.reshape(-1, feat_dim)
        inputs = x
        for idx in range(self.netdepth):
            x = self.pts_linears[idx](x)
            x = self.net_activation(x)
            if idx % self.skip_layer == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        raw_density = self.density_layer(x).reshape(
            -1, num_samples, self.num_density_channels
        )

        bottleneck = self.bottleneck_layer(x)

        x = torch.cat([bottleneck, condition], dim=-1)
        for idx in range(self.netdepth_condition):
            x = self.views_linear[idx](x)
            x = self.net_activation(x)

        raw_rgb = self.rgb_layer(x).reshape(-1, num_samples, self.num_rgb_channels)

        return raw_rgb, raw_density

@gin.configurable
class TraMNeRFRough(nn.Module):
    def __init__(
        self,
        num_levels: int = 2,
        min_deg_point: int = 0,
        max_deg_point: int = 10,
        deg_view: int = 4,
        num_coarse_samples: int = 64,
        num_fine_samples: int = 128,
        noise_std: float = 0.0,
        lindisp: bool = False,
        mirrors: list[Mirror] = None,
        num_bounces: int = 1,
        num_ray_samples: int = None,
        variance: float = 0.05,
        brdf_weight_factor: float = 1.0,
        use_linear_colors: bool = False
    ):
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(TraMNeRFRough, self).__init__()

        if self.use_linear_colors:
            self.rgb_activation = nn.Softplus()
        else:
            self.rgb_activation = nn.Sigmoid()

        self.sigma_activation = nn.ReLU()
        self.coarse_mlp = TraMNeRFRoughMLP(min_deg_point, max_deg_point, deg_view)
        self.fine_mlp = TraMNeRFRoughMLP(min_deg_point, max_deg_point, deg_view)
        self.num_bounces = num_bounces

        # convert to legacy mirror representation
        mirrors = parse_mirror_descriptors(mirrors, device='cuda:0')
        corners = torch.cat([mirrors[0].points, mirrors[1].points[1:2]], dim=0)
        normal = torch.cross(corners[1] - corners[0], corners[3] - corners[0], dim=-1)
        self.mirrors = torch.nn.ModuleList(
            get_mirror_list(
                mirror_points=[corners],
                mirror_origin=[corners.mean(dim=0)],
                mirror_normal=[normal]
            )
        )

        phi = torch.linspace(- torch.pi / 2, torch.pi / 2, 1024, device=self.mirrors[0].normal.device)
        pdf = self.variance**2 / (torch.pi*(torch.cos(phi)**2*(self.variance**2 - 1) + 1)**2)

        self.discretized_ggx_cdf = torch.cumsum(pdf * torch.pi / 1024, dim=0)
        self.discretized_ggx_cdf = self.discretized_ggx_cdf / self.discretized_ggx_cdf[-1]

    def get_network_parameters(self):
        return [*self.coarse_mlp.parameters(), *self.fine_mlp.parameters()]
    
    def forward(
        self,
        rays: Rays,
        randomized: bool,
        white_bkgd: bool,
        near: float,
        far: float,
        return_debug_images: bool = True
    ) -> dict:

        # set this to false for our approach
        # use_same_rays = False

        # patch rough model defaults for older configs
        default_names_values = (
            ("use_same_rays", False),
            ("brdf_weight_factor", 1.0),
            ("use_linear_colors", False)
        )

        for default_name, default_value in default_names_values:
            if not hasattr(self, default_name):
                setattr(self, default_name, default_value)

        ret = dict()

        for i_level in range(self.num_levels):
            # with torch.no_grad():
            if i_level == 0:
                t_vals, samples, viewdirs, monte_carlo_t, monte_carlo_samples, monte_carlo_viewdirs, reflected_rays, true_reflections, intersection_t, brdf_weights = helper.sample_along_rays(
                    rays_o=rays.origins,
                    rays_d=rays.viewdirs,
                    num_samples=self.num_coarse_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp,
                    num_bounces=self.num_bounces,
                    mirrors = self.mirrors,
                    monte_carlo_rays=self.num_ray_samples,
                    variance=self.variance,
                    discretized_ggx_cdf=self.discretized_ggx_cdf,
                    use_same_rays=self.use_same_rays
                )
                mlp = self.coarse_mlp
                viewdirs = viewdirs.reshape(-1, viewdirs.shape[-1])
                
            else:
                t_mids = 0.5 * (t_vals[..., 1:] + t_vals[..., :-1])
                t_vals, samples, viewdirs, monte_carlo_t, monte_carlo_samples, monte_carlo_viewdirs, reflected_rays, true_reflections, intersection_t, brdf_weights = helper.sample_pdf(
                    bins=t_mids,
                    weights=weights[..., 1:-1],
                    origins=rays.origins,
                    directions=rays.viewdirs,
                    t_vals=t_vals,
                    num_samples=self.num_fine_samples,
                    randomized=randomized,
                    num_bounces=self.num_bounces,
                    mirrors = self.mirrors,
                    monte_carlo_rays=self.num_ray_samples,
                    variance=self.variance,
                    discretized_ggx_cdf=self.discretized_ggx_cdf,
                    use_same_rays=self.use_same_rays
                )
                mlp = self.fine_mlp
                viewdirs = viewdirs.reshape(-1, viewdirs.shape[-1])

            samples_enc = helper.pos_enc(
                samples,
                self.min_deg_point,
                self.max_deg_point,
            )

            if monte_carlo_samples.size()[0] != 0:
                monte_carlo_viewdirs = monte_carlo_viewdirs.reshape(-1, monte_carlo_viewdirs.shape[-1])
                monte_samples_encoded = helper.pos_enc(
                    monte_carlo_samples,
                    self.min_deg_point,
                    self.max_deg_point
                )
                monte_viewdirs_enc = helper.pos_enc(monte_carlo_viewdirs, 0, self.deg_view)
                raw_monte_rgb, raw_monte_sigma = mlp(monte_samples_encoded, monte_viewdirs_enc)
                bsz = raw_monte_rgb.size()[0]


            viewdirs_enc = helper.pos_enc(viewdirs, 0, self.deg_view)
            raw_rgb, raw_sigma = mlp(samples_enc.detach(), viewdirs_enc.detach())
            samples_size = raw_rgb.size()[1]


            if self.noise_std > 0 and randomized:
                raw_sigma = raw_sigma + torch.rand_like(raw_sigma) * self.noise_std
                

            rgb = self.rgb_activation(raw_rgb)
            sigma = self.sigma_activation(raw_sigma)


            if monte_carlo_samples.size()[0] == 0:
                comp_rgb, acc, weights, _ = helper.volumetric_rendering(
                    rgb,
                    sigma,
                    t_vals,
                    rays.viewdirs,
                    white_bkgd=white_bkgd,
                )
            else:
                monte_rgb = self.rgb_activation(raw_monte_rgb).view(bsz, self.num_ray_samples, samples_size, 3)
                monte_sigma = self.sigma_activation(raw_monte_sigma).view(bsz, self.num_ray_samples, samples_size, 1)
                monte_carlo_t = monte_carlo_t.view(bsz, self.num_ray_samples, samples_size)
                monte_carlo_viewdirs = monte_carlo_viewdirs.view(bsz, self.num_ray_samples, samples_size, 3)

                absolute_monte_carlo_t = monte_carlo_t + intersection_t[reflected_rays][:, None, None]
                
                monte_carlo_weights = brdf_weights.view(bsz, self.num_ray_samples, samples_size) / self.num_ray_samples
                monte_carlo_weights = torch.where(
                    true_reflections[reflected_rays][:, None],
                    monte_carlo_weights * self.brdf_weight_factor,
                    monte_carlo_weights
                )
                
                comp_rgb, acc, weights = helper.monte_carlo_volumetric_rendering(
                    rgb,
                    sigma,
                    t_vals, #relative_t_vals,
                    monte_rgb,
                    monte_sigma,
                    absolute_monte_carlo_t,
                    monte_carlo_weights,
                    reflected_rays,
                    white_bkgd=white_bkgd,
                )
                    
            if comp_rgb.isnan().any():
                warnings.warn("Warning, there are NaN values in comp_rgb.")

            if self.use_linear_colors:
                comp_rgb = helper.linear_to_gamma(comp_rgb)

            level_name = ["coarse", "fine"][i_level]
            ret[f"{level_name}_rgb"] = comp_rgb
            ret[f"{level_name}_acc"] = acc

        return ret