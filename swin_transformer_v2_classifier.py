import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Any
import numpy as np

# Import SwinTransformerV2 model and components from repository
from swin_transformer_v2.model import SwinTransformerV2, swin_transformer_v2_b
from swin_transformer_v2.model_parts import PatchEmbedding, SwinTransformerStage

class SwinTransformerV2Classifier(nn.Module):
    """
    This class implements the Swin Transformer V2 with a classification head.
    """

    def __init__(self,
                 in_channels: int = 3,
                 embedding_channels: int = 128,
                 depths: Tuple[int, ...] = (2, 2, 18, 2),
                 input_resolution: Tuple[int, int] = (224, 224),
                 number_of_heads: Tuple[int, ...] = (4, 8, 16, 32),
                 window_size: int = 7,
                 patch_size: int = 4,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: float = 0.2,
                 use_checkpoint: bool = False,
                 sequential_self_attention: bool = False,
                 use_deformable_block: bool = False,
                 num_classes: int = 1000) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param embedding_channels: (int) Number of embedding channels
        :param depths: (Tuple[int, ...]) Depth of each stage
        :param input_resolution: (Tuple[int, int]) Input resolution
        :param number_of_heads: (Tuple[int, ...]) Number of attention heads in each stage
        :param window_size: (int) Window size
        :param patch_size: (int) Patch size
        :param ff_feature_ratio: (int) Feed-forward feature ratio
        :param dropout: (float) Dropout rate
        :param dropout_attention: (float) Attention dropout rate
        :param dropout_path: (float) Path dropout rate (Stochastic Depth)
        :param use_checkpoint: (bool) Use checkpointing
        :param sequential_self_attention: (bool) Use sequential self-attention
        :param use_deformable_block: (bool) Use deformable blocks
        :param num_classes: (int) Number of classes for classification
        """
        # Call super constructor
        super(SwinTransformerV2Classifier, self).__init__()
        
        # Initialize the backbone
        self.backbone = SwinTransformerV2(
            in_channels=in_channels,
            embedding_channels=embedding_channels,
            depths=depths,
            input_resolution=input_resolution,
            number_of_heads=number_of_heads,
            window_size=window_size,
            patch_size=patch_size,
            ff_feature_ratio=ff_feature_ratio,
            dropout=dropout,
            dropout_attention=dropout_attention,
            dropout_path=dropout_path,
            use_checkpoint=use_checkpoint,
            sequential_self_attention=sequential_self_attention,
            use_deformable_block=use_deformable_block
        )
        
        # Calculate the output channels from the last stage
        last_stage_channels = embedding_channels * 2 ** (len(depths) - 1)
        
        # Layer normalization
        self.norm = nn.LayerNorm(last_stage_channels)
        
        # 增加特徵增強層，提高特徵提取能力
        self.feature_enhance = nn.Sequential(
            nn.Linear(last_stage_channels, last_stage_channels),
            nn.GELU(),
            nn.Dropout(0.1),
        )
        
        # 使用更穩定的特徵縮放，避免NaN問題
        self.feature_scaling = nn.Sequential(
            nn.LayerNorm(last_stage_channels, elementwise_affine=False),  # 使用LayerNorm替代BatchNorm1d提高穩定性
            nn.Dropout(0.1)  # 添加額外的dropout以防止過擬合
        )
        
        # Classification head (global average pooling + linear)
        self.head = nn.Linear(last_stage_channels, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # 使用更穩定的初始化方法
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def update_resolution(self, new_window_size: int, new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the window size and input resolution
        :param new_window_size: (int) New window size
        :param new_input_resolution: (Tuple[int, int]) New input resolution
        """
        self.backbone.update_resolution(new_window_size=new_window_size, new_input_resolution=new_input_resolution)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to extract features
        :param x: (torch.Tensor) Input tensor of shape [B, C, H, W]
        :return: (torch.Tensor) Feature tensor
        """
        # Extract features from backbone
        features = self.backbone(x)
        
        # Get the last stage features
        x = features[-1]
        
        # Convert to BHWC format for layer norm
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1)  # B, H, W, C
        
        # Apply layer norm
        x = self.norm(x)
        
        # Global average pooling
        x = x.reshape(B, H * W, C).mean(dim=1)  # B, C
        
        # 增強特徵表示
        if hasattr(self, 'feature_enhance'):
            x = self.feature_enhance(x)
        
        return x
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param x: (torch.Tensor) Input tensor of shape [B, C, H, W]
        :return: (torch.Tensor) Output tensor of shape [B, num_classes]
        """
        # Extract features
        x = self.forward_features(x)
        
        # Apply feature scaling from Swin v2 paper
        # This helps stabilize the feature magnitude during training
        if hasattr(self, 'feature_scaling') and self.training:
            x = self.feature_scaling(x)

        # Classification head
        x = self.head(x)
        
        return x


def swin_transformer_v2_base_classifier(input_resolution: Tuple[int, int] = (224, 224),
                                       window_size: int = 7,
                                       in_channels: int = 3,
                                       num_classes: int = 1000,
                                       use_checkpoint: bool = False,
                                       sequential_self_attention: bool = False,
                                       dropout_path: float = 0.2) -> SwinTransformerV2Classifier:
    """
    Creates a Swin Transformer V2 Base model for classification
    :param input_resolution: (Tuple[int, int]) Input resolution
    :param window_size: (int) Window size
    :param in_channels: (int) Number of input channels
    :param num_classes: (int) Number of classes
    :param use_checkpoint: (bool) Use checkpointing
    :param sequential_self_attention: (bool) Use sequential self-attention
    :param dropout_path: (float) Stochastic depth rate
    :return: (SwinTransformerV2Classifier) SwinV2 classifier
    """
    return SwinTransformerV2Classifier(
        in_channels=in_channels,
        embedding_channels=192,
        depths=(2, 2, 18, 2),
        input_resolution=input_resolution,
        number_of_heads=(6, 12, 24, 48),
        window_size=window_size,
        dropout_path=dropout_path,  # 增加 stochastic depth 參數
        num_classes=num_classes,
        use_checkpoint=use_checkpoint,
        sequential_self_attention=sequential_self_attention
    )
