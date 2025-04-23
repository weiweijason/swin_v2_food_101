import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Union, Any, Dict, Optional
import numpy as np
import os

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
        :param dropout_path: (float) Path dropout rate
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
        
        # Classification head (global average pooling + linear)
        self.head = nn.Linear(last_stage_channels, num_classes)
        
        # Initialize weights
        self.apply(self._init_weights)

        # With this, using your model's actual last stage dimension:
        # Assuming in_features is the dimension of your final features before the classifier
        # This is typically the same value you're using for your head's input dimension
        in_features = self.backbone.num_features if hasattr(self.backbone, 'num_features') else self.head.in_features
        self.feature_scaling = nn.Sequential(
            nn.BatchNorm1d(in_features, affine=False),  
            nn.Identity()
        )
    
    def load_pretrained_weights(self, pretrained_path: str, strict: bool = False, num_classes: Optional[int] = None) -> None:
        """
        Load pretrained weights from a checkpoint file.
        
        :param pretrained_path: Path to the pretrained model checkpoint
        :param strict: Whether to strictly enforce that the keys in state_dict match the keys in model
        :param num_classes: Optional new number of classes (if None, keeps the current model's num_classes)
        """
        if not os.path.exists(pretrained_path):
            raise FileNotFoundError(f"預訓練權重檔案未找到: {pretrained_path}")
        
        print(f"載入預訓練權重: {pretrained_path}")
        
        try:
            # 載入預訓練狀態字典，使用 weights_only=True 增強安全性
            state_dict = torch.load(pretrained_path, map_location='cpu', weights_only=True)
            
            # 處理不同的狀態字典格式
            if isinstance(state_dict, dict):
                if 'model' in state_dict:
                    state_dict = state_dict['model']
                elif 'state_dict' in state_dict:
                    state_dict = state_dict['state_dict']
                # 處理 huggingface 格式
                elif any(k.startswith('backbone.') or k.startswith('encoder.') for k in state_dict.keys()):
                    # 有些預訓練權重可能有前綴
                    print("檢測到 HuggingFace 格式的預訓練權重，正在適配...")
                    # 嘗試匹配模型權重
                    new_state_dict = {}
                    for k, v in state_dict.items():
                        # 移除可能的前綴
                        if k.startswith('backbone.'):
                            new_k = k[len('backbone.'):]
                            new_state_dict[new_k] = v
                        elif k.startswith('encoder.'):
                            new_k = k[len('encoder.'):]
                            new_state_dict[new_k] = v
                        else:
                            new_state_dict[k] = v
                    state_dict = new_state_dict
            else:
                raise ValueError("載入的預訓練權重格式不正確")
                
            # 過濾掉不兼容的鍵（特別是分類頭）如果 num_classes 不同
            if num_classes is not None and num_classes != self.head.out_features:
                # 保存原始分類頭的輸出特徵
                original_out_features = self.head.out_features
                
                # 創建新的分類頭，使用所需的類別數量
                in_features = self.head.in_features
                self.head = nn.Linear(in_features, num_classes)
                
                # 從狀態字典中移除分類頭參數
                state_dict = {k: v for k, v in state_dict.items() if 'head' not in k and 'fc' not in k and 'classifier' not in k}
                
                print(f"重建分類頭: {original_out_features} -> {num_classes} 類")
            
            # 檢查 state_dict 中的權重維度是否與當前模型相符
            # 特別關注 PatchEmbed 層和相對位置偏差表
            incompatible_shapes = []
            for name, param in self.named_parameters():
                if name in state_dict:
                    if param.shape != state_dict[name].shape:
                        incompatible_shapes.append((name, param.shape, state_dict[name].shape))
            
            if incompatible_shapes:
                print("檢測到以下不兼容的參數形狀:")
                for name, current_shape, loaded_shape in incompatible_shapes:
                    print(f"  {name}: 當前 {current_shape}, 載入 {loaded_shape}")
                print("嘗試調整一部分參數...")
                
                # 移除不相容形狀的鍵
                for name, _, _ in incompatible_shapes:
                    state_dict.pop(name, None)
            
            # 載入狀態字典
            result = self.load_state_dict(state_dict, strict=strict)
            
            # 輸出缺失和多餘的鍵
            if len(result.missing_keys) > 0:
                print(f"缺失的鍵: {result.missing_keys}")
            if len(result.unexpected_keys) > 0:
                print(f"多餘的鍵: {result.unexpected_keys}")
                
            print(f"成功載入預訓練權重")
            
        except Exception as e:
            print(f"載入預訓練權重時發生錯誤: {e}")
            print("將使用隨機初始化權重繼續訓練...")
            # 權重加載失敗時，不中斷訓練，而是使用隨機初始化的權重
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
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
                                       pretrained_path: Optional[str] = None) -> SwinTransformerV2Classifier:
    """
    Creates a Swin Transformer V2 Base model for classification
    :param input_resolution: (Tuple[int, int]) Input resolution
    :param window_size: (int) Window size
    :param in_channels: (int) Number of input channels
    :param num_classes: (int) Number of classes
    :param use_checkpoint: (bool) Use checkpointing
    :param sequential_self_attention: (bool) Use sequential self-attention
    :param pretrained_path: (Optional[str]) Path to pretrained weights
    :return: (SwinTransformerV2Classifier) SwinV2 classifier
    """
    model = SwinTransformerV2Classifier(
        in_channels=in_channels,
        embedding_channels=128,
        depths=(2, 2, 18, 2),
        input_resolution=input_resolution,
        number_of_heads=(4, 8, 16, 32),
        window_size=window_size,
        num_classes=num_classes,
        use_checkpoint=use_checkpoint,
        sequential_self_attention=sequential_self_attention
    )
    
    if pretrained_path is not None:
        model.load_pretrained_weights(pretrained_path, strict=False, num_classes=num_classes)
    
    return model
