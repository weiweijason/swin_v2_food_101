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

    def __init__(self, input_resolution=(224, 224), window_size=7, embed_dim=128, depths=[2, 2, 18, 2], 
                 num_heads=[4, 8, 16, 32], mlp_ratio=4., qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1, 
                 drop_path_rate=0.1, norm_layer=nn.LayerNorm, patch_size=4, ape=False, patch_norm=True, 
                 use_checkpoint=False, num_classes=1000, pretrained=None, use_avgpool=True, dropout_path=0.2):
        """
        Swin Transformer V2 分類器模型的初始化函數

        參數:
            pretrained: 預訓練模型路徑，若為None則從頭開始訓練
            use_avgpool: 是否在backbone後使用全局平均池化
            dropout_path: 深度Dropout率，用於正則化深層網絡
        """
        super().__init__()
        self.num_classes = num_classes
        self.use_avgpool = use_avgpool
        # 是否使用DDP模式（將在前向傳播中檢測）
        self.in_ddp_mode = False
        
        # 記錄參數列表，用於DDP訓練
        self._param_list = []

        # 建立 Swin Transformer V2 骨幹
        self.backbone = SwinTransformerV2(
            input_resolution=input_resolution,
            patch_size=patch_size, 
            in_channels=3,
            embedding_channels=embed_dim,
            depths=depths, 
            number_of_heads=num_heads,
            window_size=window_size, 
            ff_feature_ratio=mlp_ratio,
            dropout=drop_rate,
            dropout_attention=attn_drop_rate,
            dropout_path=dropout_path,
            use_checkpoint=use_checkpoint
        )

        # 設置靜態圖，避免在分散式訓練中重複標記參數
        if use_checkpoint:
            # 檢查是否有 _set_static_graph 屬性，如果沒有則不嘗試設置
            if hasattr(self.backbone, '_set_static_graph'):
                self.backbone._set_static_graph(True)
                print("已啟用靜態圖模式，以解決 checkpoint 中的參數重用問題")
            else:
                print("模型不支持 _set_static_graph 方法，將使用其他方式處理 checkpoint")

        # 獲取backbone最後一層特徵的維度
        backbone_output_dim = embed_dim * 2 ** (len(depths) - 1)
        
        # 使用較複雜的分類頭，增強食物特徵提取能力
        if use_avgpool:
            # 改進分類頭，替換為更強大的分類器
            # 增加中間層、殘差連接和更多的正則化
            self.head = nn.Sequential(
                nn.AdaptiveAvgPool1d(1),
                nn.Flatten(),
                nn.Linear(backbone_output_dim, backbone_output_dim // 2),
                nn.LayerNorm(backbone_output_dim // 2),  # 使用LayerNorm代替BatchNorm提高穩定性
                nn.GELU(),  # 使用GELU激活函數
                nn.Dropout(0.3),  # 較高的dropout率
                nn.Linear(backbone_output_dim // 2, backbone_output_dim // 4),
                nn.LayerNorm(backbone_output_dim // 4),
                nn.GELU(),
                nn.Dropout(0.2),
                nn.Linear(backbone_output_dim // 4, num_classes)  # 確保最後一層輸出大小為類別數量
            )
            # 打印調試信息以確認輸出維度
            print(f"分類頭最後一層輸出大小：{num_classes}")
        else:
            self.head = nn.Identity()
            
        # 註冊參數掛鉤，用於DDP訓練
        for name, param in self.named_parameters():
            self._param_list.append(param)
        print(f"初始化完成，模型共有 {len(self._param_list)} 個參數")
        
        # 載入預訓練權重
        if pretrained is not None:
            try:
                self.load_pretrained(pretrained)
                print(f"成功載入預訓練權重: {pretrained}")
            except Exception as e:
                print(f"無法載入預訓練權重: {e}，將從頭開始訓練")

    def load_pretrained(self, pretrained_path):
        """載入預訓練權重並處理可能的不匹配"""
        checkpoint = torch.load(pretrained_path, map_location='cpu')
        
        # 檢查checkpoint格式，可能包含'model'鍵
        if 'model' in checkpoint:
            checkpoint = checkpoint['model']
        
        # 處理key不匹配的情況
        state_dict = {}
        for k, v in checkpoint.items():
            # 移除'backbone.'前綴如果有的話
            if k.startswith('backbone.'):
                k = k[9:]
            
            # 只載入backbone部分的權重
            if k.startswith('patch_embed') or k.startswith('layers') or k.startswith('norm'):
                state_dict[f'backbone.{k}'] = v
        
        # 載入權重，允許部分不匹配
        self.backbone.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        """前向傳播函數"""
        # 使用旁路網絡確保前面的層也有梯度流動
        features = self.backbone(x)
        
        # 使用最後一層特徵進行分類
        output = features[-1]
        
        # 強制所有參數參與梯度計算 - 這是解決 DDP 中未使用參數問題的關鍵
        if self.training:
            dummy_loss = 0.0
            # 直接迭代每個參數，使用更直接的方式確保梯度流動
            for name, p in self.named_parameters():
                if p.requires_grad:
                    # 使用極小數乘以參數的平均值，確保梯度可以流動但不改變結果
                    dummy_loss = dummy_loss + torch.sum(p.view(-1)[0:1]) * 0.0
            
            # 將虛擬損失添加到輸出中
            output = output + dummy_loss
        
        # 如果使用avgpool，則使用自定義頭部處理
        if self.use_avgpool:
            if output.dim() == 3:  # 處理序列輸出 (B, L, C)
                output = output.transpose(1, 2)  # 轉換為 (B, C, L)
            output = self.head(output)
        
        return output


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
    :param in_channels: (int) Number of input channels (只用於文檔，實際未使用)
    :param num_classes: (int) Number of classes
    :param use_checkpoint: (bool) Use checkpointing
    :param sequential_self_attention: (bool) Use sequential self-attention (只用於文檔，實際未使用)
    :param dropout_path: (float) Stochastic depth rate
    :return: (SwinTransformerV2Classifier) SwinV2 classifier
    """
    model = SwinTransformerV2Classifier(
        input_resolution=input_resolution,
        window_size=window_size,
        num_classes=num_classes,
        depths=(2, 2, 18, 2),
        embed_dim=96,
        num_heads=[3, 6, 12, 24],
        dropout_path=dropout_path,
        use_checkpoint=use_checkpoint,
        drop_rate=0.1
    )
    
    # 打印模型狀態
    print(f"SwinV2 Classifier 初始化完成: 圖像尺寸={input_resolution}, 窗口大小={window_size}, 類別數={num_classes}")
    
    return model
