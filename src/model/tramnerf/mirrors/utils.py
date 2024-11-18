import torch

def masked_less(data_a: torch.Tensor, data_b: torch.Tensor, mask_a: torch.Tensor, mask_b: torch.Tensor, vmax=1e10):
    return torch.where(mask_a, data_a, vmax) < torch.where(mask_b, data_b, vmax)

def get_plane_on_basis(normal: torch.Tensor, eps: float = 1e-10):
    e1 = torch.tensor([1.0, 0.0, 0.0], device=normal.device)
    u = torch.cross(normal, e1)
    
    if u.norm() < eps:
        e2 = torch.tensor([0.0, 1.0, 0.0], device=normal.device)
        u = torch.cross(normal, e2)

    u = u / u.norm()
    v = torch.cross(normal, u)
    v = v / v.norm()

    return u, v

def cylinder_to_mesh(origin: torch.Tensor, end: torch.Tensor, radius: float, resolution: int):
    device = origin.device

    direction = end - origin
    length = direction.norm()
    direction = direction / length

    center = (origin + end) / 2

    # generate upper circle
    u, v = get_plane_on_basis(direction)

    angles = torch.linspace(0, 2 * torch.pi, steps=resolution + 1, device=device)[:-1]
    circle_verts = u[None] * radius * torch.cos(angles)[:, None]
    circle_verts = circle_verts + v[None] * radius * torch.sin(angles)[:, None]
    circle_verts = circle_verts + direction[None] * length / 2

    # generate vertices as offset of upper circle
    verts = torch.cat([
        circle_verts,
        circle_verts - direction * length,
        (direction * length / 2)[None],
        (- direction * length / 2)[None]
    ], dim=0)

    # move to center
    verts = verts + center[None]

    # generate face indices
    top_side_faces = torch.stack([
        torch.arange(resolution), # [0, 1, ..., r - 1]
        torch.arange(1, resolution + 1) % resolution + resolution, # [r + 1, r + 2, ..., r]
        torch.arange(1, resolution + 1) % resolution, # [1, 2, ..., 0]
    ], dim=1)

    bottom_side_faces = torch.stack([
        torch.arange(resolution), # [0, 1, ..., r]
        torch.arange(resolution) + resolution, # [r, r + 1, ..., 2r - 1]
        torch.arange(1, resolution + 1) % resolution + resolution, # [r + 1, r + 2, ..., r]
    ], dim=1)

    top_faces = torch.stack([
        torch.arange(resolution), # [0, 1, ..., r - 1]
        torch.arange(1, resolution + 1) % resolution, # [1, 2, ..., 0]
        torch.ones(resolution) * 2 * resolution, # [2r, 2r, ...]
    ], dim=1)

    bottom_faces = torch.stack([
        torch.arange(resolution) + resolution, # [r, r + 1, ..., 2r - 1]
        torch.ones(resolution) * (2 * resolution + 1), # [2r + 1, 2r + 1, ...]
        torch.arange(1, resolution + 1) % resolution + resolution # [r + 1, r + 2, ..., r]
    ], dim=1)

    faces = torch.cat([
        top_side_faces,
        bottom_side_faces,
        top_faces,
        bottom_faces
    ], dim=0).to(device).to(torch.int32)

    return verts, faces
