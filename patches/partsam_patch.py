"""Pre-import patch — replace BOTH apex AND torch LayerNorm with manual Python impl
because ROCm's layer_norm kernel hangs indefinitely on PartSAM shapes (gfx1201)."""
import torch
import torch.nn as nn
import torch.nn.functional as F

# (1) Replace apex Fused* with stubs
import apex.normalization as _apexnorm
class _FusedLayerNormStub(nn.LayerNorm):
    def __init__(self, normalized_shape, eps=1e-5, elementwise_affine=True, memory_efficient=False, **kwargs):
        super().__init__(normalized_shape, eps=eps, elementwise_affine=elementwise_affine)

class _FusedRMSNormStub(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True, **kwargs):
        super().__init__()
        self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
        self.eps = eps
        if elementwise_affine:
            self.weight = nn.Parameter(torch.ones(self.normalized_shape))
        else:
            self.register_parameter('weight', None)
    def forward(self, x):
        v = x.float().pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(v + self.eps)
        if self.weight is not None:
            x = x * self.weight
        return x

_apexnorm.FusedLayerNorm = _FusedLayerNormStub
_apexnorm.MixedFusedLayerNorm = _FusedLayerNormStub
_apexnorm.FusedRMSNorm = _FusedRMSNormStub

# (2) MAIN FIX — replace torch.nn.LayerNorm.forward with manual computation.
# ROCm's F.layer_norm calls miopenStatusUnknownError or hangs forever on certain shapes.
def _manual_layer_norm_forward(self, x):
    mean = x.mean(-1, keepdim=True)
    var = (x - mean).pow(2).mean(-1, keepdim=True)
    x_norm = (x - mean) * torch.rsqrt(var + self.eps)
    if self.elementwise_affine:
        if self.weight is not None:
            x_norm = x_norm * self.weight
        if self.bias is not None:
            x_norm = x_norm + self.bias
    return x_norm

nn.LayerNorm.forward = _manual_layer_norm_forward
print("[patch] apex stubs + nn.LayerNorm.forward → manual Python impl (bypasses ROCm broken kernel)")
