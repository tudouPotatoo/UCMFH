import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer, LayerNorm, Dropout, Linear
from torch.nn.functional import normalize
from torch.nn import functional as F

class UnimodalTransformer(nn.Module):
    """单模态Transformer编码器，用于学习图像或文本的内部特征关系"""
    def __init__(self, d_model=512, nhead=8, num_layers=2, dim_feedforward=2048, dropout=0.1):
        super(UnimodalTransformer, self).__init__()
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = LayerNorm(d_model)
        
    def forward(self, x):
        # x: [batch_size, d_model]
        x = x.unsqueeze(1)  # [batch_size, 1, d_model] - 添加序列维度
        x = self.transformer(x)  # Transformer编码
        x = x.squeeze(1)  # [batch_size, d_model] - 移除序列维度
        return self.norm(x)


class CrossAttentionFusion(nn.Module):
    """双向Cross-Attention融合模块"""
    def __init__(self, d_model=512, nhead=8, dropout=0.1):
        super(CrossAttentionFusion, self).__init__()
        
        # Image-to-Text Cross-Attention (图像作为Query，从文本中获取信息)
        self.img2text_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Text-to-Image Cross-Attention (文本作为Query，从图像中获取信息)
        self.text2img_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer Normalization (用于残差连接后的归一化)
        self.norm_img = LayerNorm(d_model)
        self.norm_text = LayerNorm(d_model)
        
        # Dropout for residual connection
        self.dropout = Dropout(dropout)
        
    def forward(self, img_feat, text_feat):
        """
        双向Cross-Attention交互
        Args:
            img_feat: [batch_size, d_model] 图像特征
            text_feat: [batch_size, d_model] 文本特征
        Returns:
            img_fused: [batch_size, d_model] 融合后的图像特征
            text_fused: [batch_size, d_model] 融合后的文本特征
        """
        # 添加序列维度以适配MultiheadAttention
        img = img_feat.unsqueeze(1)   # [batch, 1, d_model]
        text = text_feat.unsqueeze(1) # [batch, 1, d_model]
        
        # 第一个方向：图像作为Query，文本作为Key/Value
        # 让图像特征从文本中获取互补信息
        img_attended, _ = self.img2text_attn(
            query=img,
            key=text,
            value=text
        )  # [batch, 1, d_model]
        # 残差连接 + LayerNorm
        img_fused = self.norm_img(img + self.dropout(img_attended))
        
        # 第二个方向：文本作为Query，图像作为Key/Value
        # 让文本特征从图像中获取互补信息
        text_attended, _ = self.text2img_attn(
            query=text,
            key=img,
            value=img
        )  # [batch, 1, d_model]
        # 残差连接 + LayerNorm
        text_fused = self.norm_text(text + self.dropout(text_attended))
        
        # 移除序列维度
        img_fused = img_fused.squeeze(1)   # [batch, d_model]
        text_fused = text_fused.squeeze(1) # [batch, d_model]
        
        return img_fused, text_fused


class FuseTransEncoder(nn.Module):
    """改进的双流+Cross-Attention编码器 (移除冗余的TransformerEncoder)"""
    def __init__(self,  num_layers, hidden_size, nhead):
        super(FuseTransEncoder, self).__init__()
        # 第一阶段：双流架构 单模态Transformer (学习图像/文本内部关系)
        self.image_transformer = UnimodalTransformer(d_model=512, nhead=8, num_layers=2, dropout=0.1)
        self.text_transformer = UnimodalTransformer(d_model=512, nhead=8, num_layers=2, dropout=0.1)
        
        # 第二阶段：双向Cross-Attention (替代原来的冗余TransformerEncoder)
        self.cross_attention = CrossAttentionFusion(d_model=512, nhead=8, dropout=0.1)
        
        self.d_model = hidden_size
        self.sigal_d = int(self.d_model/2)
       
    def forward(self, tokens):
        """
        前向传播：单模态增强 -> 双向Cross-Attention融合
        Args:
            tokens: [1, batch_size, 1024] 或 [batch_size, 1024]
        Returns:
            img: [batch_size, 512] 融合后的图像特征
            txt: [batch_size, 512] 融合后的文本特征
        """
        # 处理输入维度 (兼容原有代码的不同输入格式)
        if tokens.dim() == 3:
            tokens_reshaped = tokens.reshape(-1, self.d_model)  # [batch_size, 1024]
        else:
            tokens_reshaped = tokens  # [batch_size, 1024]
        
        # 分离图像和文本特征
        img_input = tokens_reshaped[:, :self.sigal_d]  # [batch_size, 512]
        txt_input = tokens_reshaped[:, self.sigal_d:]  # [batch_size, 512]
        
        # 第一阶段：单模态特征增强
        img_enhanced = self.image_transformer(img_input)  # [batch_size, 512]
        txt_enhanced = self.text_transformer(txt_input)  # [batch_size, 512]
        
        # 第二阶段：双向Cross-Attention融合 (替代原来的跨模态融合Transformer)
        # 图像从文本获取信息，文本从图像获取信息
        img_fused, txt_fused = self.cross_attention(img_enhanced, txt_enhanced)
        
        # L2归一化 (保持与原代码一致)
        img = normalize(img_fused, p=2, dim=1)  # [batch_size, 512]
        txt = normalize(txt_fused, p=2, dim=1)  # [batch_size, 512]
        
        return img, txt


class ImageMlp(nn.Module):
    """
    图像哈希映射网络 (带渐进式哈希学习)
    
    借鉴HashNet的"Deep Learning to Hash by Continuation"思想：
    - 训练初期：使用较小的scale，输出连续值（易于优化）
    - 训练中期：逐渐增大scale，输出向-1/+1靠拢
    - 训练后期：scale很大，输出几乎是二值化的
    
    这样可以：
    1. 避免直接学习离散值的困难
    2. 保持梯度流畅，训练更稳定
    3. 最终获得更好的二值哈希码
    """
    def __init__(self, input_dim=512, hash_lens=64, use_progressive=True):
        super(ImageMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, hash_lens)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.use_progressive = use_progressive
        
    def forward(self, x, scale=1.0):
        """
        Args:
            x: 输入特征 [batch_size, input_dim]
            scale: 缩放参数，控制输出的"锐度"
                  - scale=1: 输出较"软"，范围广
                  - scale=10: 输出很"硬"，接近-1或+1
        Returns:
            哈希码表示 [batch_size, hash_lens]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch_size, hash_lens]
        
        # 🆕 渐进式哈希学习
        if self.use_progressive and scale > 1.0:
            # 使用tanh将输出压缩到(-1, 1)，scale控制陡峭程度
            # scale越大，输出越接近-1或+1
            x = torch.tanh(scale * x)
        
        return normalize(x, p=2, dim=1)

class TextMlp(nn.Module):
    """
    文本哈希映射网络 (带渐进式哈希学习)
    
    同ImageMlp，支持渐进式哈希码学习
    """
    def __init__(self, input_dim=512, hash_lens=64, use_progressive=True):
        super(TextMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 1024)
        self.fc2 = nn.Linear(1024, hash_lens)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(0.3)
        self.use_progressive = use_progressive
        
    def forward(self, x, scale=1.0):
        """
        Args:
            x: 输入特征 [batch_size, input_dim]
            scale: 缩放参数，控制输出的"锐度"
        Returns:
            哈希码表示 [batch_size, hash_lens]
        """
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # [batch_size, hash_lens]
        
        # 🆕 渐进式哈希学习
        if self.use_progressive and scale > 1.0:
            x = torch.tanh(scale * x)
        
        return normalize(x, p=2, dim=1)


