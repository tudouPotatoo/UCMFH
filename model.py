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

class FuseTransEncoder(nn.Module):
    def __init__(self,  num_layers, hidden_size, nhead):
        super(FuseTransEncoder, self).__init__()
        # 双流架构：单模态Transformer
        self.image_transformer = UnimodalTransformer(d_model=512, nhead=8, num_layers=2, dropout=0.1)
        self.text_transformer = UnimodalTransformer(d_model=512, nhead=8, num_layers=2, dropout=0.1)
        
        # 跨模态融合Transformer
        encoder_layer = TransformerEncoderLayer(d_model=hidden_size, nhead=nhead, batch_first=False)
        self.transformerEncoder = TransformerEncoder(encoder_layer= encoder_layer, num_layers= num_layers)
        self.d_model = hidden_size
        self.sigal_d = int(self.d_model/2)
       
    def forward(self, tokens):
        # tokens shape: [1, batch_size, 1024]
        # 先reshape以获取图像和文本特征
        tokens_reshaped = tokens.reshape(-1, self.d_model)  # [batch_size, 1024]
        img_input = tokens_reshaped[:, :self.sigal_d]  # [batch_size, 512]
        txt_input = tokens_reshaped[:, self.sigal_d:]  # [batch_size, 512]
        
        # 第一阶段：单模态特征增强
        img_enhanced = self.image_transformer(img_input)  # [batch_size, 512]
        txt_enhanced = self.text_transformer(txt_input)  # [batch_size, 512]
        
        # 重新拼接并reshape回原始格式
        enhanced_tokens = torch.cat([img_enhanced, txt_enhanced], dim=1)  # [batch_size, 1024]
        enhanced_tokens = enhanced_tokens.unsqueeze(0)  # [1, batch_size, 1024]
        
        # 第二阶段：跨模态融合
        encoder_X = self.transformerEncoder(enhanced_tokens)
        encoder_X_r = encoder_X.reshape(-1, self.d_model)
        encoder_X_r = normalize(encoder_X_r, p=2, dim=1)
        img, txt = encoder_X_r[:, :self.sigal_d], encoder_X_r[:, self.sigal_d:]
        return img, txt


class ImageMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024,128,1024], dropout=0.1):
        super(ImageMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()
    def _ff_block(self, x):
        x = normalize(x, p =2 ,dim =1)
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid)
        return out
        
    def forward(self, X):  
        mlp_output = self._ff_block(X)
        mlp_output = normalize(mlp_output, p =2 ,dim =1)
        return mlp_output

class TextMlp(nn.Module):
    def __init__(self, input_dim, hash_lens, dim_feedforward=[1024,128,1024], dropout=0.1): 
        super(TextMlp, self).__init__()
        self.fc1 = nn.Linear(input_dim, 4096)
        self.relu = nn.ReLU(inplace=True)
        self.dp = nn.Dropout(0.3)
        self.tohash = nn.Linear(4096, hash_lens)
        self.tanh = nn.Tanh()
       
    def _ff_block(self, x):
        x = normalize(x, p =2 ,dim =1)
        feat = self.relu(self.fc1(x))
        hid = self.tohash(self.dp(feat))
        out = self.tanh(hid)
        return out
        
    def forward(self, X):  
        mlp_output =  self._ff_block(X)
        mlp_output = normalize(mlp_output, p =2 ,dim =1)
        return mlp_output


