import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from typing import Tuple, Optional, List, Union, Any

# Custom DropPath implementation to avoid timm dependency
class DropPath(nn.Module):
    """
    Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0.):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # work with diff dim tensors, not just 2D ConvNets
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()  # binarize
        output = x.div(keep_prob) * random_tensor
        return output

# Add this at the beginning of your original model_parts.py file
# Then replace timm.models.layers.DropPath with just DropPath in the SwinTransformerBlock __init__ function
