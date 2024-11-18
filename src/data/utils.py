# ------------------------------------------------------------------------------------
# Modified from NerfAcc (https://github.com/nerfstudio-project/nerfacc)
# Copyright (c) 2022 Ruilong Li, UC Berkeley.
# ------------------------------------------------------------------------------------

# import collections
import torch
from typing import NamedTuple
from tensor_shape_assert import ShapedTensor

# Rays = collections.namedtuple("Rays", ("origins", "viewdirs", "radii", "cam_idxs", "xy"))
class Rays(NamedTuple):
    origins: torch.Tensor  # ["b 3"]
    viewdirs: torch.Tensor  # ["b 3"]
    radii: torch.Tensor  # ["b 1"]
    cam_idxs: torch.Tensor  # ["b"]
    xy: torch.Tensor  # ["b 2"]

class SampledRays(NamedTuple):
    origins: torch.Tensor  # ["b r 3"]
    viewdirs: torch.Tensor  # ["b r 3"]
    radii: torch.Tensor  # ["b 1"]
    cam_idxs: torch.Tensor  # ["b"]
    xy: torch.Tensor  # ["b 2"]

def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))
