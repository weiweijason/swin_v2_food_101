from custom_model_parts import DropPath
from typing import Tuple, Optional, List, Union, Any
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
import timm

__all__: List[str] = ["SwinTransformerStage", "SwinTransformerBlock", "DeformableSwinTransformerBlock", "WindowAttention"]


class FeedForward(nn.Sequential):
    """
    Feed forward module used in the transformer encoder.
    """

    def __init__(self,
                 in_features: int,
                 hidden_features: int,
                 out_features: int,
                 dropout: float = 0.) -> None:
        """
        Constructor method
        :param in_features: (int) Number of input features
        :param hidden_features: (int) Number of hidden features
        :param out_features: (int) Number of output features
        :param dropout: (float) Dropout factor
        """
        # Call super constructor and init modules
        super().__init__(
            nn.Linear(in_features=in_features, out_features=hidden_features),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=hidden_features, out_features=out_features),
            nn.Dropout(p=dropout)
        )


def bchw_to_bhwc(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, height, width, channels]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, height, width, channels]
    """
    return input.permute(0, 2, 3, 1)


def bhwc_to_bchw(input: torch.Tensor) -> torch.Tensor:
    """
    Permutes a tensor to the shape [batch size, channels, height, width]
    :param input: (torch.Tensor) Input tensor of the shape [batch size, height, width, channels]
    :return: (torch.Tensor) Output tensor of the shape [batch size, channels, height, width]
    """
    return input.permute(0, 3, 1, 2)


def unfold(input: torch.Tensor,
           window_size: int) -> torch.Tensor:
    """
    Unfolds (non-overlapping) a given feature map by the given window size (stride = window size)
    :param input: (torch.Tensor) Input feature map of the shape [batch size, channels, height, width]
    :param window_size: (int) Window size to be applied
    :return: (torch.Tensor) Unfolded tensor of the shape [batch size * windows, channels, window size, window size]
    """
    # Get original shape
    _, channels, height, width = input.shape  
    # Unfold input
    output: torch.Tensor = input.unfold(dimension=3, size=window_size, step=window_size) \
        .unfold(dimension=2, size=window_size, step=window_size)
    # Reshape to [batch size * windows, channels, window size, window size]
    output: torch.Tensor = output.permute(0, 2, 3, 1, 5, 4).reshape(-1, channels, window_size, window_size)
    return output


def fold(input: torch.Tensor,
         window_size: int,
         height: int,
         width: int) -> torch.Tensor:
    """
    Fold a tensor of windows again to a 4D feature map
    :param input: (torch.Tensor) Input tensor of windows [batch size * windows, channels, window size, window size]
    :param window_size: (int) Window size to be reversed
    :param height: (int) Height of the feature map
    :param width: (int) Width of the feature map
    :return: (torch.Tensor) Folded output tensor of the shape [batch size, channels, height, width]
    """
    # Get channels of windows
    channels: int = input.shape[1]
    # Get original batch size
    batch_size: int = int(input.shape[0] // (height * width // window_size // window_size))
    # Reshape input to
    output: torch.Tensor = input.view(batch_size, height // window_size, width // window_size, channels,
                                      window_size, window_size)
    output: torch.Tensor = output.permute(0, 3, 1, 4, 2, 5).reshape(batch_size, channels, height, width)
    return output


def to_2tuple(x):
    """將輸入轉換為2元組"""
    if isinstance(x, tuple):
        return x
    return (x, x)


def window_partition(x, window_size):
    """
    將特徵圖分割為局部窗口
    Args:
        x: (B, H, W, C)
        window_size (int): 窗口大小
    Returns:
        windows: (num_windows*B, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    
    # 檢查輸入尺寸是否為窗口大小的整數倍，如果不是，則進行填充
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    if pad_h > 0 or pad_w > 0:
        # 進行填充以確保尺寸是窗口大小的整數倍
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
        # 更新填充後的尺寸
        _, H, W, _ = x.shape
    
    # 確保尺寸能被窗口大小整除
    assert H % window_size == 0 and W % window_size == 0, f"特徵圖尺寸 ({H}x{W}) 必須能被窗口大小 {window_size} 整除"
    
    # 將特徵圖分割為窗口
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    將局部窗口合併為特徵圖
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): 窗口大小
        H (int): 原始高度
        W (int): 原始寬度
    Returns:
        x: (B, H, W, C)
    """
    # 計算填充後的尺寸（確保是窗口大小的整數倍）
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    
    # 填充後的高度和寬度
    padded_H, padded_W = H + pad_h, W + pad_w
    
    # 計算每個維度的窗口數量
    h_windows = padded_H // window_size
    w_windows = padded_W // window_size
    
    # 計算批次大小
    B = int(windows.shape[0] / (h_windows * w_windows))
    
    # 重塑窗口為填充後的特徵圖
    x = windows.view(B, h_windows, w_windows, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    x = x.view(B, padded_H, padded_W, -1)
    
    # 如果有填充，移除填充部分
    if pad_h > 0 or pad_w > 0:
        x = x[:, :H, :W, :]
    
    return x


class Mlp(nn.Module):
    """使用兩個線性層的多層感知器"""

    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., use_bn=False):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm1d(hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        if self.use_bn:
            x = self.bn(x.transpose(1, 2)).transpose(1, 2)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class WindowAttention(nn.Module):
    """窗口多頭自注意力模塊，支持相對位置偏置和移位窗口"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, attn_drop=0., proj_drop=0., use_gate=False):
        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相對位置偏置參數表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))

        # 獲取窗口中每對位置的相對坐標索引
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        # 計算相對坐標偏差
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # 將坐標從 [-Wh+1, Wh-1] 轉換為 [0, 2*Wh-2]
        relative_coords[:, :, 1] += self.window_size[1] - 1  # 將坐標從 [-Ww+1, Ww-1] 轉換為 [0, 2*Ww-2]
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)

        # QKV投影
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # 使用門控機制
        self.use_gate = use_gate
        if use_gate:
            self.gate = nn.Parameter(torch.zeros(1, num_heads, 1, 1) + 1e-3)

        # 初始化相對位置偏置表
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x, mask=None):
        """
        Args:
            x: 輸入特徵圖，[B*nW, N, C]
            mask: (可選) 注意力遮罩，[nW, N, N]，N = window_size[0] * window_size[1]
        """
        B_, N, C = x.shape
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B_, self.num_heads, N, C // self.num_heads]

        # 計算注意力分數
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        # 相對位置偏置
        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        attn = attn + relative_position_bias.unsqueeze(0)

        # 如果有遮罩，添加到注意力分數
        if mask is not None:
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = F.softmax(attn, dim=-1)
        else:
            attn = F.softmax(attn, dim=-1)

        attn = self.attn_drop(attn)

        # 門控注意力機制
        if self.use_gate:
            attn = attn * F.sigmoid(self.gate)

        # 注意力輸出
        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class WindowMultiHeadAttention(nn.Module):
    """
    This class implements window-based Multi-Head-Attention.
    """

    def __init__(self,
                 in_features: int,
                 window_size: int,
                 number_of_heads: int,
                 dropout_attention: float = 0.,
                 dropout_projection: float = 0.,
                 meta_network_hidden_features: int = 256,
                 sequential_self_attention: bool = False) -> None:
        """
        Constructor method
        :param in_features: (int) Number of input features
        :param window_size: (int) Window size
        :param number_of_heads: (int) Number of attention heads
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_projection: (float) Dropout rate after projection
        :param meta_network_hidden_features: (int) Number of hidden features in the two layer MLP meta network
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        """
        # Call super constructor
        super(WindowMultiHeadAttention, self).__init__()
        # Check parameter
        assert (in_features % number_of_heads) == 0, \
            "The number of input features (in_features) are not divisible by the number of heads (number_of_heads)."
        # Save parameters
        self.in_features: int = in_features
        self.window_size: int = window_size
        self.number_of_heads: int = number_of_heads
        self.head_dim: int = in_features // number_of_heads
        self.sequential_self_attention: bool = sequential_self_attention
        self.scale: float = self.head_dim ** -0.5
        
        # Init query, key and value mapping as a single layer
        self.mapping_qkv: nn.Module = nn.Linear(in_features=in_features, out_features=in_features * 3, bias=True)
        # Init attention dropout
        self.attention_dropout: nn.Module = nn.Dropout(dropout_attention)
        # Init projection mapping
        self.projection: nn.Module = nn.Linear(in_features=in_features, out_features=in_features, bias=True)
        # Init projection dropout
        self.projection_dropout: nn.Module = nn.Dropout(dropout_projection)
        # Init meta network for positional encodings
        self.meta_network: nn.Module = nn.Sequential(
            nn.Linear(in_features=2, out_features=meta_network_hidden_features, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=meta_network_hidden_features, out_features=number_of_heads, bias=True))

        # Add initialization for the meta network weights
        nn.init.trunc_normal_(self.meta_network[0].weight, std=.02)
        nn.init.trunc_normal_(self.meta_network[2].weight, std=.02)
        
        # 使用對數尺度的初始化，以確保更穩定的注意力計算
        self.register_parameter("tau", torch.nn.Parameter(torch.log(10.0 * torch.ones((1, number_of_heads, 1, 1)))))
        
        # Init pair-wise relative positions (log-spaced)
        self.__make_pair_wise_relative_positions()

    def __make_pair_wise_relative_positions(self) -> None:
        """
        Method initializes the pair-wise relative positions to compute the positional biases
        """
        # 使用 arange 而不是 meshgrid 來提高效率
        coords_h = torch.arange(self.window_size, device=self.tau.device)
        coords_w = torch.arange(self.window_size, device=self.tau.device)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing="ij"))
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        
        # 計算相對位置座標
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        
        # 使用 log-space 距離
        relative_coords_log = torch.sign(relative_coords) * torch.log(1. + relative_coords.abs())
        
        self.register_buffer("relative_coords_log", relative_coords_log.view(-1, 2))

    def update_resolution(self,
                          new_window_size: int,
                          **kwargs: Any) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param kwargs: (Any) Unused
        """
        # Set new window size
        self.window_size: int = new_window_size
        # Make new pair-wise relative positions
        self.__make_pair_wise_relative_positions()

    def __get_relative_positional_encodings(self) -> torch.Tensor:
        """
        Method computes the relative positional encodings
        :return: (torch.Tensor) Relative positional encodings [1, number of heads, window size ** 2, window size ** 2]
        """
        # 使用 meta network 生成相對位置編碼
        relative_position_bias = self.meta_network(self.relative_coords_log)
        relative_position_bias = relative_position_bias.view(
            self.window_size * self.window_size, 
            self.window_size * self.window_size, 
            self.number_of_heads
        ).permute(2, 0, 1)  # nH, Wh*Ww, Wh*Ww
        
        # 增加穩定性：將位置偏差限制在適當範圍內
        relative_position_bias = 16 * torch.sigmoid(relative_position_bias)
        
        return relative_position_bias.unsqueeze(0)  # 1, nH, Wh*Ww, Wh*Ww

    def __self_attention(self,
                         query: torch.Tensor,
                         key: torch.Tensor,
                         value: torch.Tensor,
                         batch_size_windows: int,
                         tokens: int,
                         mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function performs standard (non-sequential) scaled cosine self-attention
        :param query: (torch.Tensor) Query tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param key: (torch.Tensor) Key tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param value: (torch.Tensor) Value tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param batch_size_windows: (int) Size of the first dimension of the input tensor (batch size * windows)
        :param tokens: (int) Number of tokens in the input
        :param mask: (Optional[torch.Tensor]) Attention mask for the shift case
        :return: (torch.Tensor) Output feature map of the shape [batch size * windows, tokens, channels]
        """
        # 採用更穩定的注意力計算方式
        # 計算 QK^T，使用 einsum 提高效率
        attn = torch.einsum("bhqd, bhkd -> bhqk", query, key)
        
        # 標準化注意力權重：使用 scale 代替手動計算 norm 除法
        attn = attn * self.scale
        
        # 應用 logit_scale 參數控制注意力權重的溫度
        logit_scale = torch.clamp(self.tau, max=torch.log(torch.tensor(1./0.01, device=self.tau.device))).exp()
        attn = attn * logit_scale
        
        # 應用相對位置偏差來增強位置感知能力
        attn = attn + self.__get_relative_positional_encodings()
        
        # 應用遮罩（如果需要）
        if mask is not None:
            number_of_windows: int = mask.shape[0]
            attn = attn.view(batch_size_windows // number_of_windows, number_of_windows,
                             self.number_of_heads, tokens, tokens)
            attn = attn + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.number_of_heads, tokens, tokens)
        
        # 應用 softmax 獲得注意力權重
        attn = F.softmax(attn, dim=-1)
        
        # 應用 dropout
        attn = self.attention_dropout(attn)
        
        # 應用注意力權重到 value 上
        output = torch.einsum("bhal, bhlv -> bhav", attn, value)
        output = output.permute(0, 2, 1, 3).reshape(batch_size_windows, tokens, -1)
        
        return output

    def __sequential_self_attention(self,
                                    query: torch.Tensor,
                                    key: torch.Tensor,
                                    value: torch.Tensor,
                                    batch_size_windows: int,
                                    tokens: int,
                                    mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        This function performs sequential scaled cosine self-attention
        :param query: (torch.Tensor) Query tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param key: (torch.Tensor) Key tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param value: (torch.Tensor) Value tensor of the shape [batch size * windows, heads, tokens, channels // heads]
        :param batch_size_windows: (int) Size of the first dimension of the input tensor (batch size * windows)
        :param tokens: (int) Number of tokens in the input
        :param mask: (Optional[torch.Tensor]) Attention mask for the shift case
        :return: (torch.Tensor) Output feature map of the shape [batch size * windows, tokens, channels]
        """
        # 初始化輸出張量
        output = torch.zeros_like(query)
        
        # 預先計算相對位置偏差，避免在循環中重複計算
        relative_position_bias = self.__get_relative_positional_encodings()
        
        # 計算 key 的標準化係數（預先計算）
        key_norm = torch.norm(key, dim=-1, keepdim=True).clamp(min=1e-6)
        
        # 迭代查詢和鍵值對
        for token_index_query in range(tokens):
            # 獲取當前查詢向量
            q = query[:, :, token_index_query].unsqueeze(2)  # [B*W, H, 1, D]
            
            # 計算標準化的查詢向量
            q_norm = torch.norm(q, dim=-1, keepdim=True).clamp(min=1e-6)
            
            # 計算注意力分數，使用 scaled dot-product
            attn = torch.matmul(q, key.transpose(-2, -1))  # [B*W, H, 1, T]
            attn = attn / (q_norm * key_norm.transpose(-2, -1))
            
            # 應用 scale 因子
            logit_scale = torch.clamp(self.tau, max=torch.log(torch.tensor(1./0.01, device=self.tau.device))).exp()
            attn = attn / logit_scale
            
            # 應用相對位置偏差
            attn = attn + relative_position_bias[..., token_index_query:token_index_query+1, :]
            
            # 應用遮罩（如果需要）
            if mask is not None:
                number_of_windows = mask.shape[0]
                attn_view = attn.view(batch_size_windows // number_of_windows, 
                                     number_of_windows, self.number_of_heads, 1, tokens)
                mask_view = mask.unsqueeze(1).unsqueeze(0)[..., token_index_query:token_index_query+1, :]
                attn_view = attn_view + mask_view
                attn = attn_view.view(-1, self.number_of_heads, 1, tokens)
            
            # 應用 softmax 獲得注意力權重
            attn = F.softmax(attn, dim=-1)
            
            # 應用 dropout
            attn = self.attention_dropout(attn)
            
            # 應用注意力權重到 value 上
            output[:, :, token_index_query] = torch.matmul(attn, value).squeeze(2)
        
        # 重塑輸出
        output = output.permute(0, 2, 1, 3).reshape(batch_size_windows, tokens, -1)
        
        return output

    def forward(self,
                input: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size * windows, channels, height, width]
        :param mask: (Optional[torch.Tensor]) Attention mask for the shift case
        :return: (torch.Tensor) Output tensor of the shape [batch size * windows, channels, height, width]
        """
        # 保存原始形狀
        batch_size_windows, channels, height, width = input.shape
        tokens = height * width
        
        # 將輸入重塑為 [batch_size_windows, tokens, channels]
        input = input.reshape(batch_size_windows, channels, tokens).permute(0, 2, 1)
        
        # 執行 QKV 映射
        qkv = self.mapping_qkv(input)
        
        # 重塑 QKV 以便分離 Q、K、V
        qkv = qkv.reshape(batch_size_windows, tokens, 3, self.number_of_heads, channels // self.number_of_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # 3, B*W, H, T, C/H
        
        # 分離 Q、K、V
        query, key, value = qkv[0], qkv[1], qkv[2]
        
        # 執行注意力機制
        if self.sequential_self_attention:
            output = self.__sequential_self_attention(
                query=query, key=key, value=value,
                batch_size_windows=batch_size_windows, tokens=tokens,
                mask=mask
            )
        else:
            output = self.__self_attention(
                query=query, key=key, value=value,
                batch_size_windows=batch_size_windows, tokens=tokens,
                mask=mask
            )
        
        # 執行線性映射和 dropout
        output = self.projection(output)
        output = self.projection_dropout(output)
        
        # 將輸出重塑為原始形狀 [batch_size_windows, channels, height, width]
        output = output.permute(0, 2, 1).reshape(batch_size_windows, channels, height, width)
        
        return output


class SwinTransformerBlock(nn.Module):
    """
    實現 Swin Transformer 區塊，包含窗口自注意力和 FFN
    """
    
    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        
        # 對於從頭訓練，使用 PreNorm 而非 PostNorm 可以提高穩定性
        # 修改為 PreNorm 結構，先進行標準化再進行注意力計算
        self.norm1 = norm_layer(dim)
        
        # 使用增強版本的 WindowAttention
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop,
            use_gate=True  # 添加門控機制以提高從頭訓練的效果
        )
        
        # 降低 path dropout 強度以提高從頭訓練的穩定性
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        
        # 使用修改版的FFN，加入批次歸一化以提高訓練穩定性
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, 
                      act_layer=act_layer, drop=drop, use_bn=True)  # 增加 BatchNorm

        # 提前計算注意力遮罩以加速訓練
        if self.shift_size > 0:
            # 初始化時計算注意力遮罩，避免重複計算
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, self.window_size)
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.register_buffer("attn_mask", attn_mask)
        else:
            self.register_buffer("attn_mask", None)
    
    def forward(self, x):
        # 檢查輸入張量的維度，並根據需要進行轉換
        if len(x.shape) == 4:  # 如果是 [B, C, H, W] 格式
            B, C, H, W = x.shape
            # 將輸入從 [B, C, H, W] 轉換為 [B, H*W, C]
            x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        else:  # 假設已經是 [B, L, C] 格式
            B, L, C = x.shape
            H, W = self.input_resolution
            assert L == H * W, "input feature has wrong size"
        
        shortcut = x
        x = self.norm1(x)  # PreNorm
        x = x.view(B, H, W, C)
        
        # 循環位移
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x
        
        # 分割成不重疊的窗口
        x_windows = window_partition(shifted_x, self.window_size)
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)
        
        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows, mask=self.attn_mask)
        
        # 合併窗口
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)
        
        # 移回原位置
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)
        
        # FFN，使用 PreNorm 結構，加入 BatchNorm 增強從頭訓練的效果
        x = shortcut + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        # 在訓練時使用殘差縮放，增強梯度傳遞
        if self.training:
            # 應用殘差縮放因子，通常設為 < 1.0 以防梯度爆炸 
            x = x * 0.9  # 殘差縮放因子
        
        return x


class DeformableSwinTransformerBlock(SwinTransformerBlock):
    """
    This class implements a deformable version of the Swin Transformer block.
    Inspired by: https://arxiv.org/pdf/2201.00520.pdf
    """

    def __init__(self,
                 in_channels: int,
                 input_resolution: Tuple[int, int],
                 number_of_heads: int,
                 window_size: int = 7,
                 shift_size: int = 0,
                 ff_feature_ratio: int = 4,
                 dropout: float = 0.0,
                 dropout_attention: float = 0.0,
                 dropout_path: float = 0.0,
                 sequential_self_attention: bool = False,
                 offset_downscale_factor: int = 2) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param input_resolution: (Tuple[int, int]) Input resolution
        :param number_of_heads: (int) Number of attention heads to be utilized
        :param window_size: (int) Window size to be utilized
        :param shift_size: (int) Shifting size to be used
        :param ff_feature_ratio: (int) Ratio of the hidden dimension in the FFN to the input channels
        :param dropout: (float) Dropout in input mapping
        :param dropout_attention: (float) Dropout rate of attention map
        :param dropout_path: (float) Dropout in main path
        :param sequential_self_attention: (bool) If true sequential self-attention is performed
        :param offset_downscale_factor: (int) Downscale factor of offset network
        """
        # Call super constructor
        super(DeformableSwinTransformerBlock, self).__init__(
            in_channels=in_channels,
            input_resolution=input_resolution,
            number_of_heads=number_of_heads,
            window_size=window_size,
            shift_size=shift_size,
            ff_feature_ratio=ff_feature_ratio,
            dropout=dropout,
            dropout_attention=dropout_attention,
            dropout_path=dropout_path,
            sequential_self_attention=sequential_self_attention
        )
        # Save parameter
        self.offset_downscale_factor: int = offset_downscale_factor
        self.number_of_heads: int = number_of_heads
        # Make default offsets
        self.__make_default_offsets()
        # Init offset network
        self.offset_network: nn.Module = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=5, stride=offset_downscale_factor,
                      padding=3, groups=in_channels, bias=True),
            nn.GELU(),
            nn.Conv2d(in_channels=in_channels, out_channels=2 * self.number_of_heads, kernel_size=1, stride=1,
                      padding=0, bias=True)
        )

    def __make_default_offsets(self) -> None:
        """
        Method generates the default sampling grid (inspired by kornia)
        """
        # Init x and y coordinates
        x: torch.Tensor = torch.linspace(0, self.input_resolution[1] - 1, self.input_resolution[1],
                                         device=self.window_attention.tau.device)
        y: torch.Tensor = torch.linspace(0, self.input_resolution[0] - 1, self.input_resolution[0],
                                         device=self.window_attention.tau.device)
        # Normalize coordinates to a range of [-1, 1]
        x: torch.Tensor = (x / (self.input_resolution[1] - 1) - 0.5) * 2
        y: torch.Tensor = (y / (self.input_resolution[0] - 1) - 0.5) * 2
        # Make grid [2, height, width]
        grid: torch.Tensor = torch.stack(torch.meshgrid([x, y])).transpose(1, 2)
        # Reshape grid to [1, height, width, 2]
        grid: torch.Tensor = grid.unsqueeze(dim=0).permute(0, 2, 3, 1)
        # Register in module
        self.register_buffer("default_grid", grid)

    def update_resolution(self, new_window_size: int, new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param new_input_resolution: (Tuple[int, int]) New input resolution
        """
        # Update resolution and window size
        super(DeformableSwinTransformerBlock, self).update_resolution(new_window_size=new_window_size,
                                                                      new_input_resolution=new_input_resolution)
        # Update default sampling grid
        self.__make_default_offsets()

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        # Get input shape
        batch_size, channels, height, width = input.shape
        # Compute offsets of the shape [batch size, 2, height / r, width / r]
        offsets: torch.Tensor = self.offset_network(input)
        # Upscale offsets to the shape [batch size, 2 * number of heads, height, width]
        offsets: torch.Tensor = F.interpolate(input=offsets,
                                              size=(height, width), mode="bilinear", align_corners=True)
        # Reshape offsets to [batch size, number of heads, height, width, 2]
        offsets: torch.Tensor = offsets.reshape(batch_size, -1, 2, height, width).permute(0, 1, 3, 4, 2)
        # Flatten batch size and number of heads and apply tanh
        offsets: torch.Tensor = offsets.view(-1, height, width, 2).tanh()
        # Cast offset grid to input data type
        if input.dtype != self.default_grid.dtype:
            self.default_grid = self.default_grid.type(input.dtype)
        # Construct offset grid
        offset_grid: torch.Tensor = self.default_grid.repeat_interleave(repeats=offsets.shape[0], dim=0) + offsets
        # Reshape input to [batch size * number of heads, channels / number of heads, height, width]
        input: torch.Tensor = input.view(batch_size, self.number_of_heads, channels // self.number_of_heads, height,
                                         width).flatten(start_dim=0, end_dim=1)
        # Apply sampling grid
        input_resampled: torch.Tensor = F.grid_sample(input=input, grid=offset_grid.clip(min=-1, max=1),
                                                      mode="bilinear", align_corners=True, padding_mode="reflection")
        # Reshape resampled tensor again to [batch size, channels, height, width]
        input_resampled: torch.Tensor = input_resampled.view(batch_size, channels, height, width)
        return super(DeformableSwinTransformerBlock, self).forward(input=input_resampled)


class PatchMerging(nn.Module):
    """
    This class implements the patch merging approach which is essential a strided convolution with normalization before
    """

    def __init__(self,
                 in_channels: int) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        """
        # Call super constructor
        super(PatchMerging, self).__init__()
        # Init normalization
        self.normalization: nn.Module = nn.LayerNorm(normalized_shape=4 * in_channels)
        # Init linear mapping
        self.linear_mapping: nn.Module = nn.Linear(in_features=4 * in_channels, out_features=2 * in_channels,
                                                   bias=False)

    def forward(self,
                input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, in channels, height, width] 
                      or [batch size, tokens, channels]
        :return: (torch.Tensor) Output tensor of the shape [batch size, 2 * in channels, height // 2, width // 2]
        """
        # 檢查輸入維度
        if len(input.shape) == 3:  # [B, L, C] 格式
            batch_size, L, channels = input.shape
            # 假設輸入序列是來自正方形特徵圖
            height = width = int(L ** 0.5)
            # 確保可以整除
            assert height * width == L, f"輸入序列長度 {L} 無法被解析為正方形特徵圖"
            # 重塑為 [B, C, H, W] 格式
            input = input.transpose(1, 2).reshape(batch_size, channels, height, width)
        
        # 獲取原始形狀
        batch_size, channels, height, width = input.shape  
        # 轉換為 [B, H, W, C] 格式
        input: torch.Tensor = bchw_to_bhwc(input)
        # 展開輸入
        input: torch.Tensor = input.unfold(dimension=1, size=2, step=2).unfold(dimension=2, size=2, step=2)
        input: torch.Tensor = input.reshape(batch_size, input.shape[1], input.shape[2], -1)
        # 規範化輸入
        input: torch.Tensor = self.normalization(input)
        # 執行線性映射
        output: torch.Tensor = bhwc_to_bchw(self.linear_mapping(input))
        return output


class PatchEmbedding(nn.Module):
    """
    Module embeds a given image into patch embeddings.
    """

    def __init__(self,
                 in_channels: int = 3,
                 out_channels: int = 96,
                 patch_size: int = 4) -> None:
        """
        Constructor method
        :param in_channels: (int) Number of input channels
        :param out_channels: (int) Number of output channels
        :param patch_size: (int) Patch size to be utilized
        :param image_size: (int) Image size to be used
        """
        # Call super constructor
        super(PatchEmbedding, self).__init__()
        # Save parameters
        self.out_channels: int = out_channels
        # Init linear embedding as a convolution
        self.linear_embedding: nn.Module = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                                     kernel_size=(patch_size, patch_size),
                                                     stride=(patch_size, patch_size))
        # Init layer normalization
        self.normalization: nn.Module = nn.LayerNorm(normalized_shape=out_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass transforms a given batch of images into a patch embedding
        :param input: (torch.Tensor) Input images of the shape [batch size, in channels, height, width]
        :return: (torch.Tensor) Patch embedding of the shape [batch size, patches + 1, out channels]
        """
        # Perform linear embedding
        embedding: torch.Tensor = self.linear_embedding(input)
        # Perform normalization
        embedding: torch.Tensor = bhwc_to_bchw(self.normalization(bchw_to_bhwc(embedding)))
        return embedding


class SwinTransformerStage(nn.Module):
    """
    This class implements a stage of the Swin transformer including multiple layers.
    """

    def __init__(self,
                 dim: int,
                 depth: int,
                 downscale: bool,
                 input_resolution: Tuple[int, int],
                 num_heads: int,
                 window_size: int = 7,
                 mlp_ratio: int = 4,
                 drop: float = 0.0,
                 attn_drop: float = 0.0,
                 dropout_path: Union[List[float], float] = 0.0,
                 use_checkpoint: bool = False,
                 use_deformable_block: bool = False) -> None:
        """
        Constructor method
        :param dim: (int) Number of input channels
        :param depth: (int) Depth of the stage (number of layers)
        :param downscale: (bool) If true input is downsampled (see Fig. 3 or V1 paper)
        :param input_resolution: (Tuple[int, int]) Input resolution
        :param num_heads: (int) Number of attention heads to be utilized
        :param window_size: (int) Window size to be utilized
        :param mlp_ratio: (int) Ratio of the hidden dimension in the FFN to the input channels
        :param drop: (float) Dropout in input mapping
        :param attn_drop: (float) Dropout rate of attention map
        :param dropout_path: (float) Dropout in main path
        :param use_checkpoint: (bool) If true checkpointing is utilized
        :param use_deformable_block: (bool) If true deformable block is used
        """
        # Call super constructor
        super(SwinTransformerStage, self).__init__()
        # Save parameters
        self.use_checkpoint: bool = use_checkpoint
        self.downscale: bool = downscale
        # Init downsampling
        self.downsample: nn.Module = PatchMerging(in_channels=dim) if downscale else nn.Identity()
        # Update resolution and channels
        self.input_resolution: Tuple[int, int] = (input_resolution[0] // 2, input_resolution[1] // 2) \
            if downscale else input_resolution
        dim = dim * 2 if downscale else dim
        # Get block
        block = DeformableSwinTransformerBlock if use_deformable_block else SwinTransformerBlock
        # Init blocks
        self.blocks: nn.ModuleList = nn.ModuleList([
            block(dim=dim,  # 修改參數名稱：in_channels -> dim
                  input_resolution=self.input_resolution,
                  num_heads=num_heads,  # 修改參數名稱：number_of_heads -> num_heads
                  window_size=window_size,
                  shift_size=0 if ((index % 2) == 0) else window_size // 2,
                  mlp_ratio=mlp_ratio,  # 修改參數名稱：ff_feature_ratio -> mlp_ratio
                  drop=drop,  # 修改參數名稱：dropout -> drop
                  attn_drop=attn_drop,  # 修改參數名稱：dropout_attention -> attn_drop
                  drop_path=dropout_path[index] if isinstance(dropout_path, list) else dropout_path)
            for index in range(depth)])

    def update_resolution(self, new_window_size: int, new_input_resolution: Tuple[int, int]) -> None:
        """
        Method updates the window size and so the pair-wise relative positions
        :param new_window_size: (int) New window size
        :param new_input_resolution: (Tuple[int, int]) New input resolution
        """
        # Update resolution
        self.input_resolution: Tuple[int, int] = (new_input_resolution[0] // 2, new_input_resolution[1] // 2) \
            if self.downscale else new_input_resolution
        # Update resolution of each block
        for block in self.blocks:  # type: SwinTransformerBlock
            block.update_resolution(new_window_size=new_window_size, new_input_resolution=self.input_resolution)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        :param input: (torch.Tensor) Input tensor of the shape [batch size, channels, height, width]
        :return: (torch.Tensor) Output tensor of the shape [batch size, 2 * channels, height // 2, width // 2]
        """
        # Downscale input tensor
        output: torch.Tensor = self.downsample(input)
        # Forward pass of each block
        for block in self.blocks:  # type: nn.Module
            # Perform checkpointing if utilized
            if self.use_checkpoint:
                output: torch.Tensor = checkpoint.checkpoint(block, output)
            else:
                output: torch.Tensor = block(output)
        return output
