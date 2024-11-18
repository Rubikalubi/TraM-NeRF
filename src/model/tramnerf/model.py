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
from src.data.utils import Rays
from .utils import Linear2sRGB

from .mirrors.base import Mirror
from .mirrors.materials import Material, OpaqueMirrorMaterial
import src.model.tramnerf.helper as helper

from tensor_shape_assert import ShapedTensor, check_tensor_shapes


@gin.configurable()
class TraMNeRFMLP(nn.Module):
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

        super(TraMNeRFMLP, self).__init__()

        self.net_activation = nn.ReLU()
        pos_size = ((max_deg_point - min_deg_point) * 2) * input_ch
        view_pos_size = (deg_view * 2 + 1) * input_ch_view
        init_layer = nn.Linear(pos_size, netwidth)
        init.xavier_uniform_(init_layer.weight)
        pts_linear = [init_layer]

        for idx in range(netdepth - 1):
            if idx % skip_layer == 0 and idx > 0:
                module = nn.Linear(netwidth + pos_size, netwidth)
            else:
                module = nn.Linear(netwidth, netwidth)
            init.xavier_uniform_(module.weight)
            pts_linear.append(module)

        self.pts_linears = nn.ModuleList(pts_linear)

        views_linear = [nn.Linear(netwidth + view_pos_size, netwidth_condition)]
        for idx in range(netdepth_condition - 1):
            layer = nn.Linear(netwidth_condition, netwidth_condition)
            init.xavier_uniform_(layer.weight)
            views_linear.append(layer)

        self.views_linear = nn.ModuleList(views_linear)

        self.bottleneck_layer = nn.Linear(netwidth, netwidth)
        self.density_layer = nn.Linear(netwidth, num_density_channels)
        self.rgb_layer = nn.Linear(netwidth_condition, num_rgb_channels)

        init.xavier_uniform_(self.bottleneck_layer.weight)
        init.xavier_uniform_(self.density_layer.weight)
        init.xavier_uniform_(self.rgb_layer.weight)

    @check_tensor_shapes()
    def forward_features(
            self,
            x: ShapedTensor["b r c_enc"],
    ) -> ShapedTensor["b r c_feat"]:
        
        batch_size, num_samples, feat_dim = x.shape
        x = x.reshape(-1, feat_dim)
        inputs = x
        for idx in range(self.netdepth):
            x = self.pts_linears[idx](x)
            x = self.net_activation(x)
            if idx % self.skip_layer == 0 and idx > 0:
                x = torch.cat([x, inputs], dim=-1)

        return x.view(batch_size, num_samples, -1)

    @check_tensor_shapes()
    def forward_density(self, x: ShapedTensor["b r c_feat"]) -> ShapedTensor["b r c_density"]:
        num_samples, feat_dim = x.shape[1:]
        return self.density_layer(x).reshape(
            -1, num_samples, self.num_density_channels
        )
    
    @check_tensor_shapes()
    def forward_rgb(
            self,
            x: ShapedTensor["b r c_feat"],
            condition: ShapedTensor["b r c_view_enc"]
    ) -> ShapedTensor["b r c_rgb"]:
        
        num_samples, feat_dim = x.shape[1:]
        bottleneck = self.bottleneck_layer(x)
        x = torch.cat([bottleneck, condition], dim=-1)
        for idx in range(self.netdepth_condition):
            x = self.views_linear[idx](x)
            x = self.net_activation(x)

        return self.rgb_layer(x).reshape(-1, num_samples, self.num_rgb_channels)

    @check_tensor_shapes()
    def forward(
            self,
            x: ShapedTensor["b r c_pos_enc"],
            condition: ShapedTensor["b r c_view_enc"]
    ) -> tuple[ShapedTensor["b r c_rgb"], ShapedTensor["b r c_density"]]:

        x = self.forward_features(x=x)
        raw_density = self.forward_density(x=x)
        raw_rgb = self.forward_rgb(x=x, condition=condition)

        return raw_rgb, raw_density

@gin.configurable()
class TraMNeRF(nn.Module):
    def __init__(
        self,
        num_samples: int = 128,
        num_ray_samples: int = 1,
        num_levels: int = 2,
        resample_padding: float = 0.01,
        stop_level_grad: bool = True,
        lindisp: bool = False,
        min_deg_point: int = 0,
        max_deg_point: int = 16,
        deg_view: int = 4,
        density_noise: float = 0,
        density_bias: float = -1,
        rgb_padding: float = 0.001,
        mirrors: list[helper.Mirror] = [],
        materials: list[Material] = [],
        mirror_scale: float = 1.0,
        mirror_center: list[float] = None,
        num_bounces: int = 4,
        use_linear_colors: bool = False
    ):
        # Layers
        for name, value in vars().items():
            if name not in ["self", "__class__"]:
                setattr(self, name, value)

        super(TraMNeRF, self).__init__()
        self.density_activation = nn.Softplus()

        if self.use_linear_colors:
            self.rgb_activation = nn.Softplus()
            self.color_mapping_fn = Linear2sRGB()
        else:
            self.rgb_activation = nn.Sigmoid()
            self.color_mapping_fn = nn.Identity()

        self.mlp = TraMNeRFMLP(min_deg_point, max_deg_point, deg_view)

        if mirror_center is None:
            mirror_center = [0.0, 0.0, 0.0]
        mirror_center = torch.tensor(mirror_center, device='cuda:0')

        self.mirrors = helper.parse_mirror_descriptors(mirrors, device='cuda:0')
        self.mirrors = [m.transformed(mirror_center, mirror_scale) for m in self.mirrors]
        self.materials = helper.parse_material_descriptors(materials)

        if len(self.materials) == 0:
            self.materials = [OpaqueMirrorMaterial()]
            warnings.warn("No materials were given in the config. Using OpaqueMirrorMaterial for "
                          "backwards compatibility.")

        self.num_bounces = num_bounces

    def get_network_parameters(self):
        return list(self.mlp.parameters())

    def forward_functional(
            self,
            rays: Rays,
            randomized: bool,
            white_bkgd: bool,
            near: float,
            far: float,
            return_debug_images: bool = True,
            num_ray_samples: int = None,
            mirrors: list[Mirror] = None
    ):
        ret = dict()

        if num_ray_samples is None:
            num_ray_samples = self.num_ray_samples
    
        for i_level in range(self.num_levels):
            
            level_name = ["coarse", "fine"][i_level]

            if i_level == 0:
                t_vals, samples, viewdirs, sample_weights, intersections, reflected_viewdirs, ray_weights, debug_tensors = helper.sample_along_rays(
                    rays=rays,
                    num_samples=self.num_samples,
                    near=near,
                    far=far,
                    randomized=randomized,
                    lindisp=self.lindisp,
                    num_bounces=self.num_bounces,
                    num_ray_samples=num_ray_samples,
                    mirrors=mirrors,
                    materials=self.materials,
                    return_debug_images=return_debug_images,
                )
            else:
                t_vals, samples, viewdirs, sample_weights, _ = helper.resample_along_rays(
                    rays=rays,
                    intersections=intersections,
                    reflected_viewdirs=reflected_viewdirs,
                    ray_weights=ray_weights,
                    t_vals=t_vals,
                    weights=weights,
                    randomized=randomized,
                    stop_level_grad=self.stop_level_grad,
                    resample_padding=self.resample_padding,
                    return_debug_images=return_debug_images,
                )

            # flatten r, s dims for a moment
            num_batch_samples, num_ray_samples = samples[0].shape[:2]
            samples_flat = (
                samples[0].view(-1, self.num_samples, 3),
                samples[1].view(-1, self.num_samples, 3)
            )
            t_vals_flat = t_vals.view(-1, self.num_samples + 1)
            weights_flat = sample_weights.view(-1, self.num_samples)
            viewdirs_flat = viewdirs.reshape(-1, self.num_samples, viewdirs.shape[-1])

            samples_flat_enc = helper.integrated_pos_enc(
                samples=samples_flat, min_deg=self.min_deg_point, max_deg=self.max_deg_point
            )

            viewdirs_flat_enc = helper.pos_enc(
                viewdirs_flat, min_deg=0, max_deg=self.deg_view, append_identity=True
            )

            # only sample rays that have non-zero weight
            raw_rgb_flat = torch.zeros_like(viewdirs_flat)
            raw_density_flat = torch.zeros_like(viewdirs_flat[:, :, :1])
            ray_weights_mask = ray_weights[0].view(-1) > 0
            raw_rgb_flat_masked, raw_density_flat_masked = self.mlp(x=samples_flat_enc[ray_weights_mask], condition=viewdirs_flat_enc[ray_weights_mask])
            raw_rgb_flat[ray_weights_mask] = raw_rgb_flat_masked
            raw_density_flat[ray_weights_mask] = raw_density_flat_masked

            if randomized and (self.density_noise > 0):
                raw_density_flat += self.density_noise * torch.rand_like(raw_density_flat)

            rgb = self.rgb_activation(raw_rgb_flat)
            rgb = rgb * (1 + 2 * self.rgb_padding) - self.rgb_padding

            density = self.density_activation(raw_density_flat + self.density_bias)[..., 0]
            expanded_viewdirs = rays.viewdirs[:, None].expand(-1, num_ray_samples, -1)#.reshape(-1, 3)

            comp_rgb_rays, distance, acc, weights = helper.volumetric_rendering(
                rgb=rgb,
                density=density,
                t_vals=t_vals_flat,
                dirs=expanded_viewdirs.reshape(-1, 3),
                sample_weights=torch.ones_like(weights_flat),
                white_bkgd=white_bkgd
            )

            # reduce ray dimension
            comp_rgb = (comp_rgb_rays.view(num_batch_samples, num_ray_samples, 3) * ray_weights[0, :, :, None]).sum(dim=1)
            comp_rgb_detached = (comp_rgb_rays.view(num_batch_samples, num_ray_samples, 3).detach() * ray_weights[0, :, :, None]).sum(dim=1)
            distance = distance.view(num_batch_samples, num_ray_samples).mean(dim=1)
            acc = acc.view(num_batch_samples, num_ray_samples).mean(dim=1)

            comp_rgb = self.color_mapping_fn(comp_rgb)

            # unflatten weights
            weights = weights.view(num_batch_samples, num_ray_samples, self.num_samples)

            # fill return dict            
            ret[f"{level_name}_rgb"] = comp_rgb
            ret[f"{level_name}_rgb_detached"] = comp_rgb_detached
            ret[f"{level_name}_dist"] = distance
            ret[f"{level_name}_acc"] = acc

            for k, v in debug_tensors.items():
                ret[f"{level_name}_{k}"] = v

        return ret
        

    def forward(
        self,
        rays: Rays,
        randomized: bool,
        white_bkgd: bool,
        near: float,
        far: float,
        return_debug_images: bool = True,
        num_ray_samples: int = None,
    ) -> dict:
        
        return self.forward_functional(
            rays=rays,
            randomized=randomized,
            white_bkgd=white_bkgd,
            near=near,
            far=far,
            return_debug_images=return_debug_images,
            num_ray_samples=num_ray_samples,
            mirrors=self.mirrors
        )