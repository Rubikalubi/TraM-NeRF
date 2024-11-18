import torch
import math

def expand(input : torch.Tensor, shape : tuple) -> torch.Tensor:
    r"""Expand singleton dimensions to a larger size. See torch.expand.
    This method allows for the number of specified dimensions in the `shape` tuple to be less than the dimensions in the `input` tensor.
    In that case, the shape is extended from the left with -1 values.

    Args:
        input (torch.Tensor): Arbitrary input tensor.
        shape: Tuple of sizes for each dimension, a subset or superset of dimensions from the input tensor

    Returns:
        torch.Tensor: View into the input tensor with expanded shape.

    Example:
        >>> input = torch.rand(2, 3, 4, 5, 1)
        >>> input.shape
        torch.Size([2, 3, 1])
        >>> expand(input, (4,))
        torch.Size([2, 3, 4])
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(input)))
    if input.dim() > len(shape):
        count = input.dim() - len(shape)
        shape = (*(-1,)*count, *shape)
    return input.expand(shape)


def dot(a : torch.Tensor, b : torch.Tensor, out=None, keepdim=False) -> torch.Tensor:
    return torch.sum(a * b, out=out, keepdim=keepdim, dim=-1)

def length(dir, keepdim=False):
    return torch.linalg.norm(dir, dim=-1, keepdim=keepdim)

def normalize(dir, eps=1e-6):
    len = length(dir, keepdim=True)
    return torch.where(len > eps, dir / len, dir)

def matvecmul(A : torch.Tensor, b : torch.Tensor) -> torch.Tensor:
    r"""Compute the bached matrix vector product between A and b.
    Treat A as an array of matrices, and b as an array of column vectors.
    A[..., 0, :] and b must be broadcastable.

    Args:
        A (torch.Tensor): Array of matrices with shape = (..., n, m).
        b (torch.Tensor): Array of vectors with shape = (..., m).

    Returns:
        torch.Tensor: Array of output vectors with shape = (..., n).
    """
    if not torch.is_tensor(A):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(A)))
    if not torch.is_tensor(b):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(b)))
    if A.dim() < 2:
        raise ValueError("Input tensor has insufficient shape. Got {}".format(
            A.dim()))
    return (A @ b[..., None])[..., 0]

def build_local_frame(localZ : torch.Tensor) -> torch.Tensor:

    x  = localZ[..., 0]
    y  = localZ[..., 1]
    z  = localZ[..., 2]
    sz = torch.where(z < 0, -1, 1).type(z.dtype)
    a  = 1 / (sz + z)
    ya = y * a
    b  = x * ya
    c  = x * sz

    return torch.stack([
        c*x*a - 1, b, x,
        sz*b, y*ya - sz, y,
        c, y, z
    ], dim=-1).view(*x.shape, 3, 3)

def build_local_frame_view_normal(view_dir : torch.Tensor, normal : torch.Tensor) -> torch.Tensor:
    NdotV = dot(view_dir, normal, keepdim=True)
    localZ = normal
    localX = normalize(view_dir - NdotV * normal)
    localY = torch.cross(localZ, localX, dim=-1)
    mask = expand((NdotV > 1 - 1e-6).view(*NdotV.shape[:-1], 1, 1), (3, 3))
    return torch.where(mask, build_local_frame(normal), torch.stack([
        localX,
        localY,
        localZ
    ], dim=-1))

def reflect(dir : torch.Tensor, normal : torch.Tensor) -> torch.Tensor:
    IoN = dot(dir, normal, keepdim=True)
    return dir - 2*IoN*normal

def sample_hemisphere_ggx_vndf(uv, view_dir, roughness):
    # return halfway
    # uv.shape = (..., 2)
    # light_dir.shape = (..., 3)
    # roughness.shape = (..., 1)
    # halfway.shape = (..., 3)
    # pdf.shape = (..., 1)

    normal_dir = torch.tensor([[0, 0, 1]], device=view_dir.device, dtype=view_dir.dtype) # shape=(1,3)

    # A transforms from ellipse to hemisphere: ve = A*vh / || A*vh ||
    Adiag = torch.cat([roughness, roughness, torch.ones_like(roughness)], dim=-1)
    # transform from ellipse to hemisphere!
    view_dir_hemisphere = normalize(Adiag * view_dir)

    # custom coordinate system on hemisphere
    frame = build_local_frame_view_normal(*torch.broadcast_tensors(normal_dir, view_dir_hemisphere))
    # view_dir -> z axis (vh)
    # normal   -> x axis (t2)

    # sample disk
    u = uv[..., 0:1]
    v = uv[..., 1:2]
    r   = torch.sqrt(u)
    phi = 2*math.pi * v
    t1 = r*torch.cos(phi)
    t2 = r*torch.sin(phi)

    # apply scaling
    NdotV = view_dir_hemisphere[..., -1:]
    s = 0.5*(1 + NdotV)
    t2 = (1-s) * torch.sqrt(1-t1**2) + s*t2
    vh = torch.sqrt((1 - t1**2 - t2**2).clamp(0, 1))
    # construct halfway in reparameterized coordinate system
    halfway_dir = torch.cat(torch.broadcast_tensors(t2, t1, vh), dim=-1) # NOTE: t1 and t2 are swapped here!
    # transform to hemisphere
    halfway_dir = matvecmul(frame, halfway_dir)
    # transform from hemisphere to ellipse
    halfway_dir = normalize(Adiag * halfway_dir)
    return halfway_dir

def F_Schlick(F0 : torch.Tensor, cosTheta : torch.Tensor) -> torch.Tensor:
    d = 1 - cosTheta
    return F0 + (1 - F0) * (d**5)

def V_SmithGGX(NdotL : torch.Tensor, NdotV : torch.Tensor, roughness : torch.Tensor, eps=1e-8) -> torch.Tensor:
    a2 = roughness**2
    lambdaV = torch.abs(NdotL) * torch.sqrt(torch.square(NdotV) * (1 - a2) + a2)
    lambdaL = torch.abs(NdotV) * torch.sqrt(torch.square(NdotL) * (1 - a2) + a2)
    return 0.5 / (lambdaV + lambdaL + eps)

def Lambda_SmithGGX(NdotV : torch.Tensor, roughness : torch.Tensor) -> torch.Tensor:
    return 0.5*(torch.sqrt(1 + roughness**2 * (1-NdotV**2) / NdotV**2) - 1)

def G1_SmithGGX(NdotV : torch.Tensor, roughness : torch.Tensor, eps=1e-8) -> torch.Tensor:
    LambdaV = Lambda_SmithGGX(NdotV, roughness)
    return 1 / (1 + LambdaV)

def G2_SmithGGX(NdotL : torch.Tensor, NdotV : torch.Tensor, roughness : torch.Tensor, eps=1e-8) -> torch.Tensor:
    LambdaV = Lambda_SmithGGX(NdotV, roughness)
    LambdaL = Lambda_SmithGGX(NdotL, roughness)
    return 1 / (1 + LambdaV + LambdaL)

def BRDF_GGX(
        normals : torch.Tensor,
        light_dirs : torch.Tensor,
        view_dirs : torch.Tensor,
        roughness : torch.Tensor,
        F0 : torch.Tensor = 1,
        include_nol=True
):
    halfway = normalize(light_dirs + view_dirs)
    NdotL = dot(normals, light_dirs, keepdim=True)
    NdotV = dot(normals, view_dirs, keepdim=True)
    NdotH = dot(normals, halfway, keepdim=True)
    HdotV = dot(halfway, view_dirs, keepdim=True)

    V = V_SmithGGX(NdotL, NdotV, roughness)
    F = F_Schlick(F0, HdotV)

    result = F * G2_SmithGGX(NdotL, NdotV, roughness) / G1_SmithGGX(NdotV, roughness)
    if not include_nol:
        result = result / NdotL
    include_nol = False


    if include_nol:
        result = torch.maximum(NdotL, torch.zeros_like(NdotL)) * result
    else:
        result = torch.where(NdotL > 0, result, torch.zeros_like(result))


    return result

def sample_brdf_ggx_vndf(uv, view_dir, roughness, F0=1, include_nol=True):
    normal_dir = torch.tensor([[0, 0, 1]], device=view_dir.device, dtype=view_dir.dtype) # shape=(1,3)
    halfway_dir = sample_hemisphere_ggx_vndf(uv, view_dir, roughness)
    light_dir = reflect(-view_dir, halfway_dir)
    brdf = BRDF_GGX(normals=normal_dir, light_dirs=light_dir, view_dirs=view_dir, roughness=roughness, F0=F0, include_nol=include_nol)
    return light_dir, brdf