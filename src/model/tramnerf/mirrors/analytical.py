import torch

from src.data.utils import Rays
from .base import Mirror, Intersection
from .utils import masked_less
from ..utils import batched_dot
from math import log

from tensor_shape_assert import ShapedTensor


class TriangleMirror(Mirror):
    def __init__(
            self,
            points: torch.Tensor,
            double_sided: bool = False,
            flip_normal: bool = False,
    ) -> None:
        super().__init__()
        
        if flip_normal:
            points = points[(0, 2, 1), :]

        self.points = points
        self.double_sided = double_sided

    def parameters(self) -> tuple[torch.Tensor]:
        return [self.points]

    def transformed(self, center: torch.Tensor, scale: float) -> Mirror:
        return TriangleMirror((self.points - center) * scale, self.double_sided)
    
    def get_intersections(
            self,
            rays: Rays,
            t_mids: ShapedTensor["n r"] = None,
        ) -> tuple[Intersection, dict[str, torch.Tensor]]:

        # Moeller-Trumbore criterion
        v_ccw = self.points[2] - self.points[1] #v0v1
        v_cw = self.points[0] - self.points[1] #v0v2

        p_vec = torch.cross(rays.viewdirs, v_cw[None], dim=-1)
        t_vec = rays.origins - self.points[1][None]
        q_vec = torch.cross(t_vec, v_ccw[None], dim=-1)

        det = batched_dot(v_ccw[None], p_vec)[:, 0, 0]
        u = batched_dot(t_vec, p_vec) / det
        v = batched_dot(rays.viewdirs, q_vec) / det

        ray_inside = (u >= 0) & (u <= 1) & (v >= 0) & ((u + v) <= 1)
        ray_is_not_parallel = torch.abs(det) > 0

        is_valid = ray_inside & ray_is_not_parallel

        if not getattr(self, "double_sided", False):
            ray_hits_front = det > 0
            is_valid = is_valid & ray_hits_front

        # compute intersection
        normal = torch.cross(v_ccw, v_cw, dim=-1)
        normal = normal / normal.norm()
        
        D = normal @ self.points[1]
        t = (-(batched_dot(normal[None], rays.origins) - D) / (batched_dot(normal[None], rays.viewdirs)))[:, 0, 0]

        if getattr(self, "double_sided", False):
            signed_normal = normal[None] * torch.sign(det)[:, None]
        else:
            signed_normal = normal.expand_as(rays.viewdirs)

        intersection_world = rays.origins + t[:, None] * rays.viewdirs

        return Intersection(
            t=t,
            world=intersection_world,
            normal=signed_normal,
            is_valid=is_valid
        ), {}

class CircleMirror(Mirror):
    def __init__(
            self,
            center: torch.Tensor,
            normal: torch.Tensor,
            radius: float
    ) -> None:
        super().__init__()
        self.center = center
        self.normal = normal
        self.radius = radius

    def parameters(self) -> tuple[torch.Tensor]:
        return [self.center, self.normal, self.radius]

    def transformed(self, center: torch.Tensor, scale: float) -> Mirror:
        return CircleMirror(
            center=(self.center - center) * scale,
            normal=self.normal,
            radius=self.radius * scale
        )
    
    def get_intersections(
            self,
            rays: Rays,
            t_mids: ShapedTensor["n r"] = None,
    ) -> tuple[Intersection, dict[str, torch.Tensor]]:
        # compute transformation into local coords
        z = self.normal
        x = torch.cross(z[None], rays.viewdirs, dim=-1)  # (B, 3)
        x = x / torch.linalg.norm(x, dim=-1, keepdim=True)
        y = torch.cross(z[None], x, dim=-1) # (B, 3)
        y = y / torch.linalg.norm(y, dim=-1, keepdim=True)

        R_to_glob = torch.stack([x, y, z.expand_as(x)], dim=-1)
        R_to_loc = R_to_glob.transpose(-1, -2) # (B, 3, 3)

        # transform origins and directions
        rays_o_local = (R_to_loc @ (rays.origins - self.center[None])[:, :, None])[:, :, 0]  # (B, 3)
        rays_d_local = (R_to_loc @ rays.viewdirs[:, :, None])[:, :, 0]  # (B, 3)

        # compute intersections with tangent plane
        t = -rays_o_local[:, 2] / rays_d_local[:, 2]
        intersect_local = torch.stack([
            rays_o_local[:, 0] + t * rays_d_local[:, 0],
            rays_o_local[:, 1] + t * rays_d_local[:, 1],
            torch.zeros_like(t)
        ], dim=1)

        # check if intersections are inside circle
        is_intersect_inside_circle = (intersect_local[:, 0] ** 2 + intersect_local[:, 1] ** 2) < self.radius ** 2
        normal_points_to_camera = batched_dot(self.normal[None], rays.viewdirs)[:, 0, 0] < 0
        is_valid = is_intersect_inside_circle & normal_points_to_camera

        # transform back to world coords
        intersect_world = (R_to_glob @ intersect_local[:, :, None])[:, :, 0] + self.center[None]

        return Intersection(
            t=t,
            world=intersect_world,
            normal=self.normal.expand_as(intersect_world),
            is_valid=is_valid
        ), {}
 
class OpenCylinderMirror(Mirror):
    def __init__(
            self,
            origin: torch.Tensor,
            end: torch.Tensor,
            radius: torch.Tensor
    ) -> None:
        super().__init__()
        self.origin = origin
        self.end = end
        self.radius = radius

    def parameters(self) -> tuple[torch.Tensor]:
        return [self.origin, self.end, self.radius]

    def transformed(self, center: torch.Tensor, scale: float) -> Mirror:
        return OpenCylinderMirror(
            origin=(self.origin - center) * scale,
            end=(self.end - center) * scale,
            radius=self.radius * scale
        )
    
    def get_intersections(
            self,
            rays: Rays,
            t_mids: ShapedTensor["n r"] = None,
    ) -> tuple[Intersection, dict[str, torch.Tensor]]:
        num_rays = rays.origins.shape[0]
        assert rays.viewdirs.shape == (num_rays, 3)

        # compute cylinder transformations, one for each ray
        # choose coordinate system as
        
        # z = lateral axis
        z = (self.end - self.origin)[None] # (1, 3)
        z = z / torch.linalg.norm(z, dim=-1, keepdim=True)

        # x = normal of the (z, ray direction) plane 
        x = torch.cross(z, rays.viewdirs, dim=-1)  # (B, 3)
        x = x / torch.linalg.norm(x, dim=-1, keepdim=True)

        # y = normal of the (z, x) plane
        y = torch.cross(z, x, dim=-1) # (B, 3)
        y = y / torch.linalg.norm(y, dim=-1, keepdim=True)

        assert z.isfinite().all(), f"{self.origin=} {self.end=} {z=}"
        assert x.isfinite().all()
        assert y.isfinite().all()

        # define rotation matrices
        R_to_glob = torch.stack([x, y, z.expand(x.shape[0], 3)], dim=-1)
        R_to_loc = R_to_glob.transpose(-1, -2) # (B, 3, 3)
        
        # transform ray origins and direction to the local coordinate system
        # s.t. cylinder origin is (0, 0, 0)
        # x, y are the radial axes of the cylinder
        # z is the lateral axis
        rays_o_local = (R_to_loc @ (rays.origins - self.origin[None])[:, :, None])[:, :, 0]  # (B, 3)
        rays_d_local = (R_to_loc @ rays.viewdirs[:, :, None])[:, :, 0]  # (B, 3)
        length = torch.linalg.norm(self.end - self.origin)
        
        # analytically solve for ||o + t * d||^2 = r^2 in the local coordindates
        r = self.radius
        a = rays_o_local[:, 0]
        b = rays_d_local[:, 0]
        c = rays_o_local[:, 1]
        d = rays_d_local[:, 1]
        
        b2_minus_d2 = (b ** 2 + d ** 2)
        ab_plus_cd = a * b + c * d
        sqrt_input = d ** 2 * (r ** 2 - a ** 2) + 2 * a * b * c * d + b ** 2 * (r ** 2 - c ** 2)
        negative_sqrt = sqrt_input < 0
        safe_term = torch.where(negative_sqrt, 1.0, sqrt_input)
        sqrt = torch.sqrt(safe_term)

        t1 = (-sqrt - ab_plus_cd) / b2_minus_d2 # (B,)
        t2 = (sqrt - ab_plus_cd) / b2_minus_d2 # (B,)

        assert sqrt.isfinite().all()
        assert t1.isfinite().all()
        assert t2.isfinite().all()
        
        # compute local intersection points
        intersect_local_1 = rays_o_local + t1[:, None] * rays_d_local # (B, 3)
        intersect_local_2 = rays_o_local + t2[:, None] * rays_d_local # (B, 3)

        # check validity
        is_intersect_1_valid = (0 < intersect_local_1[:, 2]) & (intersect_local_1[:, 2] < length) & ~negative_sqrt
        is_intersect_2_valid = (0 < intersect_local_2[:, 2]) & (intersect_local_2[:, 2] < length) & ~negative_sqrt

        is_valid = is_intersect_1_valid | is_intersect_2_valid

        # get first intersection point + normal
        t = torch.where(masked_less(t1, t2, is_intersect_1_valid, is_intersect_2_valid), t1, t2)
        intersect_local = rays_o_local + t[:, None] * rays_d_local # (B, 3)
        normal_local = intersect_local * torch.tensor([[1.0, 1.0, 0.0]], device=intersect_local.device) # (B, 3)
        normal_local = normal_local / torch.linalg.norm(normal_local, dim=-1, keepdim=True)

        assert normal_local.isfinite().all()
        
        # transform back to world coordinates
        intersect_world = (R_to_glob @ intersect_local[:, :, None])[:, :, 0] + self.origin[None]
        normal = (R_to_glob @ normal_local[:, :, None])[:, :, 0]
        

        # check if normal points towards ray
        normal_points_to_camera = batched_dot(normal, rays.viewdirs) < 0

        is_valid = is_valid & normal_points_to_camera

        return Intersection(
            t=t,
            world=intersect_world,
            normal=normal,
            is_valid=is_valid
        ), {}
    
        
    
class CylinderMirror(Mirror):
    def __init__(
            self,
            origin: torch.Tensor,
            end: torch.Tensor,
            radius: torch.Tensor
    ) -> None:
        super().__init__()
        self.origin = origin
        self.end = end
        self.radius = radius

    def transformed(self, center: torch.Tensor, scale: float) -> Mirror:
        return CylinderMirror(
            origin=(self.origin - center) * scale,
            end=(self.end - center) * scale,
            radius=self.radius * scale
        )
    
    def get_intersections(
            self, *args, **kwargs
    ) -> tuple[Intersection, dict[str, torch.Tensor]]:
        axis_direction = self.end - self.origin
        axis_direction = axis_direction / axis_direction.norm()
        
        open_cylinder = OpenCylinderMirror(
            origin=self.origin,
            end=self.end,
            radius=self.radius
        )
        origin_cap = CircleMirror(
            center=self.origin,
            normal=-axis_direction,
            radius=self.radius
        )
        end_cap = CircleMirror(
            center=self.end,
            normal=axis_direction,
            radius=self.radius
        )

        return (
            open_cylinder.get_intersections(*args, **kwargs)[0].get_closest(
            origin_cap.get_intersections(*args, **kwargs)[0]).get_closest(
            end_cap.get_intersections(*args, **kwargs)[0])
        ), {}
    