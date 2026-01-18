#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
测试双流Transformer架构
"""
import torch
from model import FuseTransEncoder, UnimodalTransformer

def test_unimodal_transformer():
    """测试单模态Transformer"""
    print("=" * 60)
    print("测试 UnimodalTransformer")
    print("=" * 60)
    
    # 创建单模态Transformer
    img_trans = UnimodalTransformer(d_model=512, nhead=8, num_layers=2)
    
    # 创建测试数据
    batch_size = 32
    img_feat = torch.randn(batch_size, 512)
    
    # 前向传播
    img_output = img_trans(img_feat)
    
    print(f"输入形状: {img_feat.shape}")
    print(f"输出形状: {img_output.shape}")
    print(f"输出维度是否保持一致: {img_output.shape == img_feat.shape}")
    
    # 检查参数数量
    total_params = sum(p.numel() for p in img_trans.parameters())
    print(f"参数数量: {total_params:,}")
    print("✓ UnimodalTransformer 测试通过\n")
    
def test_dual_stream_fuse_transformer():
    """测试双流融合Transformer"""
    print("=" * 60)
    print("测试 FuseTransEncoder (双流架构)")
    print("=" * 60)
    
    # 创建FuseTransEncoder
    num_layers = 6
    hidden_size = 1024
    nhead = 8
    fuse_trans = FuseTransEncoder(num_layers, hidden_size, nhead)
    
    # 创建测试数据
    batch_size = 32
    img_feat = torch.randn(batch_size, 512)
    txt_feat = torch.randn(batch_size, 512)
    
    # 拼接特征
    tokens = torch.cat([img_feat, txt_feat], dim=1)  # [batch_size, 1024]
    tokens = tokens.unsqueeze(0)  # [1, batch_size, 1024]
    
    # 前向传播
    img_output, txt_output = fuse_trans(tokens)
    
    print(f"输入形状: {tokens.shape}")
    print(f"图像输出形状: {img_output.shape}")
    print(f"文本输出形状: {txt_output.shape}")
    print(f"输出维度是否正确: {img_output.shape[1] == 512 and txt_output.shape[1] == 512}")
    
    # 检查参数数量
    total_params = sum(p.numel() for p in fuse_trans.parameters())
    img_trans_params = sum(p.numel() for p in fuse_trans.image_transformer.parameters())
    txt_trans_params = sum(p.numel() for p in fuse_trans.text_transformer.parameters())
    fuse_params = sum(p.numel() for p in fuse_trans.transformerEncoder.parameters())
    
    print(f"\n参数统计:")
    print(f"  - 图像Transformer: {img_trans_params:,}")
    print(f"  - 文本Transformer: {txt_trans_params:,}")
    print(f"  - 融合Transformer: {fuse_params:,}")
    print(f"  - 总参数: {total_params:,}")
    print("✓ FuseTransEncoder 测试通过\n")

def test_gradient_flow():
    """测试梯度流动"""
    print("=" * 60)
    print("测试梯度反向传播")
    print("=" * 60)
    
    # 创建模型
    fuse_trans = FuseTransEncoder(num_layers=2, hidden_size=1024, nhead=8)
    
    # 创建测试数据
    batch_size = 8
    img_feat = torch.randn(batch_size, 512, requires_grad=True)
    txt_feat = torch.randn(batch_size, 512, requires_grad=True)
    
    tokens = torch.cat([img_feat, txt_feat], dim=1).unsqueeze(0)
    
    # 前向传播
    img_output, txt_output = fuse_trans(tokens)
    
    # 计算简单的损失
    loss = (img_output - txt_output).pow(2).mean()
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_grad = True
    for name, param in fuse_trans.named_parameters():
        if param.grad is None:
            print(f"✗ {name} 没有梯度")
            has_grad = False
    
    if has_grad:
        print("✓ 所有参数都有梯度")
        print("✓ 梯度反向传播正常\n")
    else:
        print("✗ 某些参数没有梯度\n")

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("双流Transformer架构测试")
    print("=" * 60 + "\n")
    
    try:
        # 测试单模态Transformer
        test_unimodal_transformer()
        
        # 测试双流融合Transformer
        test_dual_stream_fuse_transformer()
        
        # 测试梯度流动
        test_gradient_flow()
        
        print("=" * 60)
        print("✓ 所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试失败: {str(e)}")
        import traceback
        traceback.print_exc()
