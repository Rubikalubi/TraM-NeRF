import numpy as np
import torch
from tensor_shape_assert import check_tensor_shapes, ShapedTensor, get_shape_variables

def bdot(a, b):
    # batched dot product over last dimension
    return (a[..., None, :] @ b[..., None])[..., 0, 0]

def batched_dot(a, b):
  #For the edge case of only 1 ray hitting the mirror, we don't have to use squeeze
  if a.size()[0] == 1:
    return(a[..., None, :] @ b[..., :, None])
  return (a[..., None, :] @ b[..., :, None]).squeeze()

def reflect(v, n, o=None):
    # reflects vector v at plane through o with normal n
    v = v if o is None else v - o
    v = v - 2 * bdot(v, n)[..., None] * n
    return v if o is None else v + o

@check_tensor_shapes()
def rotate(
    v: ShapedTensor["...B 3"],
    axis: ShapedTensor["...B 3"],
    cos_angle: ShapedTensor["...B"],
    o=None
) -> ShapedTensor["...B 3"]:
    # rotates vector v around axis with given cos(angle), o being center of rotation
    v = v if o is None else v - o
    cos_angle = cos_angle[..., None]
    sin_angle = torch.sqrt(1 - cos_angle ** 2)
    v =  v * cos_angle + torch.cross(axis, v, dim=-1) * sin_angle + axis * bdot(axis, v)[..., None] * (1 - cos_angle)
    return v if o is None else v + o

@check_tensor_shapes()
def get_axis_cos_angle(
    a: ShapedTensor["...B 3"],
    b: ShapedTensor["...B 3"]
) -> tuple[
   ShapedTensor["...B 3"],
   ShapedTensor["...B"]
]:
    # get rotation axis and cos(angle) to rotate a onto b
    axis = torch.cross(a, b, dim=-1)
    axis = axis / torch.linalg.norm(axis, dim=-1, keepdims=True)
    norm_a = a / a.norm(dim=-1, keepdim=True)
    norm_b = b / b.norm(dim=-1, keepdim=True)
    cos_angle = bdot(norm_a, norm_b)
    return axis, cos_angle

@check_tensor_shapes()
def rotate_a_to_b(
    v: ShapedTensor["...B 3"],
    a: ShapedTensor["...B 3"],
    b: ShapedTensor["...B 3"]
) -> ShapedTensor["...B 3"]:
    # rotates v such that a is rotated onto b
    if torch.isclose(a, b).all():
        return v
    else:
        axis, cos_angle = get_axis_cos_angle(a=a, b=b)
        return rotate(v=v, axis=axis, cos_angle=cos_angle)
    

def img2mse(x, y, mask):
    if mask is None:
        return torch.mean((x - y) ** 2)
    else:
        return torch.sum((x - y) ** 2 * mask) / mask.sum()


def mse2psnr(x):
    return -10.0 * torch.log(x) / np.log(10)

class Linear2sRGB(torch.nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(self, linear: torch.Tensor) -> torch.Tensor:
        # We clamp to min = 1e-10 to avoid nans in grad of linear.pow(1/2.4) path 
        # ( which appears to be propagated through torch.where )
        linear = torch.clamp(linear, min=1e-10, max=1.0)
        srgb = torch.where(
            linear <= 0.0031308,
            12.92 * linear,
            1.055 * linear.pow(1.0 / 2.4) - 0.055
        )
        #assert srgb.isfinite().all()
        return srgb


@check_tensor_shapes(experimental_enable_autogen_constraints=True)
def compute_transmittance(
    t_vals: ShapedTensor["b s+1"],
    density: ShapedTensor["b s"],
    dirs: ShapedTensor["b 3"] = None,
) -> tuple[ShapedTensor["b s"], ShapedTensor["b s"]]:
    t_dists = t_vals[..., 1:] - t_vals[..., :-1]

    if dirs is not None:
        delta = t_dists * torch.norm(dirs[..., None, :], dim=-1)
    else:
        delta = t_dists

    # Note that we're quietly turning density from [..., 0] to [...].
    density_delta = density * delta

    alpha = 1 - torch.exp(-density_delta)
    trans = torch.exp(
        -torch.cat(
            [
                torch.zeros_like(density_delta[..., :1]),
                torch.cumsum(density_delta[..., :-1], axis=-1),
            ],
            axis=-1,
        )
    )

    return trans, alpha
