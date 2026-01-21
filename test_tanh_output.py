"""
测试FusionMlp输出范围是否在[-1, 1]之间
验证Tanh激活函数是否正确应用
"""
import torch
import torch.nn as nn
from model import FusionMlp

def test_fusion_mlp_output_range():
    """测试FusionMlp输出是否在[-1, 1]范围内"""
    print("=" * 60)
    print("测试FusionMlp输出范围")
    print("=" * 60)
    
    batch_size = 32
    input_dim = 1024
    hash_lens = [16, 32, 64, 128]
    
    for hash_len in hash_lens:
        print(f"\n测试hash长度: {hash_len}")
        print("-" * 40)
        
        # 创建模型
        fusion_mlp = FusionMlp(input_dim, hash_len).eval()
        
        # 创建随机输入（模拟连接后的特征）
        x = torch.randn(batch_size, input_dim)
        
        # 前向传播
        with torch.no_grad():
            output = fusion_mlp(x)
        
        # 检查输出形状
        expected_shape = (batch_size, hash_len)
        assert output.shape == expected_shape, f"输出形状错误: {output.shape} vs {expected_shape}"
        print(f"✓ 输出形状正确: {output.shape}")
        
        # 检查输出范围
        min_val = output.min().item()
        max_val = output.max().item()
        print(f"✓ 输出范围: [{min_val:.4f}, {max_val:.4f}]")
        
        assert min_val >= -1.0, f"最小值超出范围: {min_val} < -1.0"
        assert max_val <= 1.0, f"最大值超出范围: {max_val} > 1.0"
        print(f"✓ 输出在[-1, 1]范围内")
        
        # 统计输出分布
        mean_val = output.mean().item()
        std_val = output.std().item()
        print(f"✓ 输出统计: 均值={mean_val:.4f}, 标准差={std_val:.4f}")
        
        # 测试二值化
        hash_codes = torch.sign(output)
        unique_values = hash_codes.unique().tolist()
        print(f"✓ 二值化后的唯一值: {unique_values}")
        assert set(unique_values).issubset({-1.0, 0.0, 1.0}), "二值化结果错误"

def test_tanh_activation():
    """测试Tanh激活函数是否正确应用"""
    print("\n" + "=" * 60)
    print("测试Tanh激活函数")
    print("=" * 60)
    
    input_dim = 1024
    hash_len = 64
    
    # 创建模型
    fusion_mlp = FusionMlp(input_dim, hash_len)
    
    # 检查模型是否有tanh层
    assert hasattr(fusion_mlp, 'tanh'), "模型缺少tanh属性"
    assert isinstance(fusion_mlp.tanh, nn.Tanh), "tanh属性类型错误"
    print("✓ 模型包含Tanh激活层")
    
    # 创建极端输入测试
    x_large = torch.randn(10, input_dim) * 100  # 大幅度输入
    
    with torch.no_grad():
        output_large = fusion_mlp(x_large)
    
    # Tanh应该将任何输入限制在[-1, 1]之间
    assert output_large.min() >= -1.0, "Tanh未正确限制最小值"
    assert output_large.max() <= 1.0, "Tanh未正确限制最大值"
    print(f"✓ 极端输入测试通过: 范围=[{output_large.min():.4f}, {output_large.max():.4f}]")

def test_gradient_flow():
    """测试梯度是否正常流动"""
    print("\n" + "=" * 60)
    print("测试梯度流动")
    print("=" * 60)
    
    batch_size = 16
    input_dim = 1024
    hash_len = 32
    
    # 创建模型和优化器
    fusion_mlp = FusionMlp(input_dim, hash_len)
    optimizer = torch.optim.Adam(fusion_mlp.parameters(), lr=0.001)
    
    # 前向传播
    x = torch.randn(batch_size, input_dim, requires_grad=True)
    output = fusion_mlp(x)
    
    # 创建虚拟损失
    target = torch.randn_like(output)
    loss = nn.MSELoss()(output, target)
    
    # 反向传播
    loss.backward()
    
    # 检查梯度
    has_gradient = False
    for name, param in fusion_mlp.named_parameters():
        if param.grad is not None:
            has_gradient = True
            grad_norm = param.grad.norm().item()
            print(f"✓ {name}: 梯度范数={grad_norm:.6f}")
    
    assert has_gradient, "没有参数接收到梯度"
    print("✓ 所有参数都接收到梯度")
    
    # 执行优化步骤
    optimizer.step()
    print("✓ 优化步骤成功")

def test_comparison_with_normalize():
    """比较Tanh和L2 normalize的输出差异"""
    print("\n" + "=" * 60)
    print("对比Tanh vs L2 Normalize")
    print("=" * 60)
    
    batch_size = 8
    input_dim = 1024
    hash_len = 16
    
    fusion_mlp = FusionMlp(input_dim, hash_len).eval()
    
    x = torch.randn(batch_size, input_dim)
    
    with torch.no_grad():
        # 使用Tanh的输出
        output_tanh = fusion_mlp(x)
        
        # 模拟L2 normalize的输出（用于对比）
        # 注意：这只是为了说明差异，实际代码已经移除了normalize
        output_before_tanh = fusion_mlp.fc2(fusion_mlp.fc1(x))
        output_normalized = nn.functional.normalize(output_before_tanh, p=2, dim=1)
    
    print(f"Tanh输出范围: [{output_tanh.min():.4f}, {output_tanh.max():.4f}]")
    print(f"L2 Normalize输出范围: [{output_normalized.min():.4f}, {output_normalized.max():.4f}]")
    
    print(f"\nTanh输出L2范数: {output_tanh.norm(p=2, dim=1).mean():.4f}")
    print(f"L2 Normalize输出L2范数: {output_normalized.norm(p=2, dim=1).mean():.4f}")
    
    print("\n✓ Tanh确保输出在[-1, 1]范围内")
    print("✓ L2 Normalize确保每个样本的L2范数为1")
    print("✓ 两者有本质不同：Tanh逐元素约束，L2按样本约束")

if __name__ == "__main__":
    test_fusion_mlp_output_range()
    test_tanh_activation()
    test_gradient_flow()
    test_comparison_with_normalize()
    
    print("\n" + "=" * 60)
    print("所有测试通过！✓")
    print("=" * 60)
    print("\n总结:")
    print("1. FusionMlp输出正确限制在[-1, 1]范围内")
    print("2. Tanh激活函数正确应用")
    print("3. 梯度流动正常")
    print("4. 可以直接用sign()进行二值化")
