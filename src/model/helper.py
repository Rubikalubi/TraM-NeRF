# ------------------------------------------------------------------------------------
# Modified from NeRF-Factory (https://github.com/kakaobrain/nerf-factory)
# Copyright (c) 2022 POSTECH, KAIST, Kakao Brain Corp. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------
# ------------------------------------------------------------------------------------
# Modified from NeRF (https://github.com/bmild/nerf)
# Copyright (c) 2020 Google LLC. All Rights Reserved.
# ------------------------------------------------------------------------------------


from typing import Callable, Generic, TypeVar
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

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


def clamp_cast_uint8(x: torch.Tensor, maximum: float = 1.0):
    if isinstance(x, torch.Tensor):
        return (x.clip(0, maximum) / maximum * 255).to(torch.uint8)
    else:
        return (np.clip(x, 0, maximum) / maximum * 255).astype(np.uint8)

def directions_to_rgb(x: torch.Tensor):
    return 0.5 * x + 0.5

def depth_to_rgb(depth: torch.Tensor, near: float, far: float):
    norm_depth = (depth - near) / (far - near)
    return torch.from_numpy(plt.cm.turbo(norm_depth.cpu().numpy()))

from functools import wraps
from tqdm import tqdm
from typing import Protocol, Any

class BatchGenerator(Protocol):
    def __call__(self, batch_size: int, batched_kwargs: dict[str, Any]) -> dict[str, Any]:
        pass

def slice_batch_generator(batch_size: int, batched_kwargs: dict[str, Any]):
    n = next(iter(batched_kwargs.values())).shape[0]
    for i in range(0, n, batch_size):
        yield {k: v[i: i + batch_size] for k, v in batched_kwargs.items()}

def batchify(
        batched_keys: list[str],
        batch_size: int,
        batch_generator: BatchGenerator = slice_batch_generator,
        output_device: str = None,
        check_shapes: bool = True,
        show_progress: bool = False
):
    def _batcher(fn):
        @wraps(fn)
        def _wrapper(*args, **kwargs):
            if len(args) > 0:
                raise ValueError("Only keyword arguments are allowed.")
            
            # check if all batch kwargs have same 0-th dimension
            n = kwargs[batched_keys[0]].shape[0]
            if check_shapes and any(kwargs[k].shape[0] != n for k in batched_keys):
                shape_dict = {k: kwargs[k].shape for k in batched_keys}
                raise ValueError(
                    f"All batch tensors must have same size in dimension 0, but "
                    f"got {shape_dict}."
                )

            # prepare kwargs and output list
            batched_kwargs = {k: v for k, v in kwargs.items() if k in batched_keys}
            other_kwargs = {k: v for k, v in kwargs.items() if k not in batched_keys}
            output_tuples = []

            # run fn on batches and collect outputs
            with tqdm(total=n, disable=not show_progress) as pbar:
                for batch in batch_generator(batch_size, batched_kwargs):
                    result = fn(**batch, **other_kwargs)

                    # create a result tuple if single return value
                    if not isinstance(result, tuple):
                        result = (result,)

                    # move to output device
                    if output_device is not None:
                        result = tuple(t.to(output_device) for t in result)

                    # save for batching
                    output_tuples.append(result)
                    pbar.update(len(result[0]))

            # batch outputs
            num_output_tensors = len(output_tuples[0])
            output_tensors = [torch.cat([o[i] for o in output_tuples], dim=0) for i in range(num_output_tensors)]

            # remove tuple if only one tensor is returned
            if len(output_tensors) == 1:
                return output_tensors[0]
            else:
                return output_tensors
            
        return _wrapper
    return _batcher


T = TypeVar("T")

class FunctionRegistry(Generic[T]):
    def __init__(self) -> None:
        self._registry: dict[str, T] = dict()

    def __getitem__(self, name: str) -> T:
        try:
            return self._registry[name]
        except KeyError:
            raise KeyError(
                f"Function key '{name}' not found in registry. Registered "
                f"functions are {list(self._registry)}."
            )
    
    def register(self, name: str | None = None):
        def _register(fn):
            _name = fn.__name__ if name is None else name
            self._registry[_name] = fn
            return fn
        return _register
