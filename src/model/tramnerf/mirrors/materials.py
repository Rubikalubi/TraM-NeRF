import torch
from ..utils import rotate_a_to_b
from tensor_shape_assert import check_tensor_shapes, ShapedTensor



class Material:
    def __init__(self, exact_num_rays: int | None = None, check_num_rays: bool = True) -> None:
        self.exact_num_rays = exact_num_rays
        self.check_num_rays = check_num_rays
    
    def sample_reflected_viewdirs_local(
            self,
            viewdirs_local: ShapedTensor["b r 3"]
    ) -> tuple[
        ShapedTensor["b r 3"],
        ShapedTensor["b r"]
    ]:
        raise NotImplementedError
    
    def parameters(self) -> list[torch.Tensor]:
        raise NotImplementedError
    
    def enable_optimization(self):
        for param in self.parameters():
            param.requires_grad = True
    
    @check_tensor_shapes()
    def sample_reflected_viewdirs(
            self,
            viewdirs: ShapedTensor["b r 3"],
            normals: ShapedTensor["b r 3"],
        ) -> tuple[
        ShapedTensor["b r 3"],
        ShapedTensor["b r"]
    ]:
        # legacy patch
        if not hasattr(self, "check_num_rays"):
            self.check_num_rays = self.exact_num_rays is not None
            
        if (
                self.check_num_rays
                and self.exact_num_rays is not None
                and viewdirs.shape[1] != self.exact_num_rays
        ):
            raise ValueError(
                f"Input tensor must have exactly r={self.exact_num_rays} number of "
                f"rays for this material to be applicable, but was {viewdirs.shape[1]}."
            )
        
        z = torch.tensor([[[0.0, 0.0, 1.0]]], device=normals.device).expand_as(normals)
        viewdirs_local = rotate_a_to_b(
            v=viewdirs,
            a=normals,
            b=z
        )
        reflected_viewdirs_local, weights = self.sample_reflected_viewdirs_local(
            viewdirs_local=viewdirs_local,
        )
        reflected_viewdirs = rotate_a_to_b(
            v=reflected_viewdirs_local,
            a=z,
            b=normals
        )

        return reflected_viewdirs, weights

class OpaqueMirrorMaterial(Material):
    def __init__(self) -> None:
        super().__init__(exact_num_rays=1)

    def parameters(self) -> list[torch.Tensor]:
        return []

    @check_tensor_shapes()
    def sample_reflected_viewdirs_local(
            self,
            viewdirs_local: ShapedTensor["b r 3"],
        ) -> tuple[
        ShapedTensor["b r 3"],
        ShapedTensor["b r"]
    ]:
        # reflect
        reflected_viewdirs = viewdirs_local.clone()
        reflected_viewdirs[:, :, 2] = -reflected_viewdirs[:, :, 2]

        # set weight to 1
        weights = torch.ones_like(viewdirs_local[:, :, 0])
        return reflected_viewdirs, weights
