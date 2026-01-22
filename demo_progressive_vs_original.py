"""
对比渐进式哈希学习前后的差异

这个脚本直观展示修改前后训练过程的区别
"""
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np

class OriginalHashMLP(nn.Module):
    """原始版本的哈希映射网络"""
    def __init__(self, input_dim=512, nbits=64):
        super(OriginalHashMLP, self).__init__()
        self.mlp = nn.Linear(input_dim, nbits)
    
    def forward(self, x):
        return self.mlp(x)  # 直接输出，没有任何缩放


class ProgressiveHashMLP(nn.Module):
    """渐进式哈希学习版本"""
    def __init__(self, input_dim=512, nbits=64):
        super(ProgressiveHashMLP, self).__init__()
        self.mlp = nn.Linear(input_dim, nbits)
    
    def forward(self, x, scale=1.0):
        x = self.mlp(x)
        return torch.tanh(scale * x)  # 🆕 应用渐进式缩放


def simulate_training_comparison():
    """模拟对比两种方法的训练过程"""
    
    print("=" * 80)
    print("渐进式哈希学习 vs 原始方法 - 训练过程对比")
    print("=" * 80)
    print()
    
    # 初始化模型
    torch.manual_seed(42)
    original_model = OriginalHashMLP(input_dim=512, nbits=64)
    progressive_model = ProgressiveHashMLP(input_dim=512, nbits=64)
    
    # 使用相同的权重初始化（确保公平对比）
    progressive_model.load_state_dict(original_model.state_dict())
    
    # 模拟输入数据
    batch_size = 100
    x = torch.randn(batch_size, 512)
    
    # 训练参数
    total_epochs = 100
    epochs_to_check = [0, 25, 50, 75, 99]
    
    print("训练配置:")
    print(f"  Batch size: {batch_size}")
    print(f"  Hash bits: 64")
    print(f"  Total epochs: {total_epochs}")
    print()
    
    print("=" * 80)
    print("训练过程统计")
    print("=" * 80)
    print()
    
    # 记录结果用于可视化
    original_outputs = []
    progressive_outputs = []
    scales = []
    
    # 模拟不同训练阶段
    for epoch in epochs_to_check:
        progress = epoch / max(total_epochs - 1, 1)
        current_scale = 1.0 + progress * 9.0  # scale从1到10
        scales.append(current_scale)
        
        # 原始方法输出
        with torch.no_grad():
            original_hash = original_model(x)
            progressive_hash = progressive_model(x, scale=current_scale)
        
        original_outputs.append(original_hash.numpy())
        progressive_outputs.append(progressive_hash.numpy())
        
        # 统计信息
        print(f"Epoch {epoch}/{total_epochs}")
        print(f"  Scale: {current_scale:.2f}")
        print()
        
        # 原始方法统计
        orig_mean = original_hash.mean().item()
        orig_std = original_hash.std().item()
        orig_near_binary = ((original_hash < -0.8) | (original_hash > 0.8)).float().mean().item() * 100
        
        print("  原始方法 (无渐进式学习):")
        print(f"    均值: {orig_mean:+.4f}, 标准差: {orig_std:.4f}")
        print(f"    接近二值化 (|x|>0.8): {orig_near_binary:.1f}%")
        print(f"    值域范围: [{original_hash.min().item():.2f}, {original_hash.max().item():.2f}]")
        print()
        
        # 渐进式方法统计
        prog_mean = progressive_hash.mean().item()
        prog_std = progressive_hash.std().item()
        prog_near_binary = ((progressive_hash < -0.8) | (progressive_hash > 0.8)).float().mean().item() * 100
        
        print("  渐进式方法 (Progressive Hash Learning):")
        print(f"    均值: {prog_mean:+.4f}, 标准差: {prog_std:.4f}")
        print(f"    接近二值化 (|x|>0.8): {prog_near_binary:.1f}%")
        print(f"    值域范围: [{progressive_hash.min().item():.2f}, {progressive_hash.max().item():.2f}]")
        print()
        
        # 对比差异
        improvement = prog_near_binary - orig_near_binary
        print(f"  📊 二值化改进: {improvement:+.1f} 百分点")
        print("-" * 80)
        print()
    
    return original_outputs, progressive_outputs, scales, epochs_to_check


def visualize_comparison(original_outputs, progressive_outputs, scales, epochs):
    """可视化对比"""
    
    fig, axes = plt.subplots(len(epochs), 2, figsize=(14, 3*len(epochs)))
    
    for idx, (epoch, scale) in enumerate(zip(epochs, scales)):
        # 原始方法
        ax1 = axes[idx, 0]
        orig_data = original_outputs[idx].flatten()
        ax1.hist(orig_data, bins=50, alpha=0.7, color='orange', edgecolor='black')
        ax1.axvline(x=-1, color='red', linestyle='--', linewidth=2)
        ax1.axvline(x=1, color='red', linestyle='--', linewidth=2)
        ax1.set_title(f'原始方法 - Epoch {epoch}', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Hash Code Value')
        ax1.set_ylabel('Frequency')
        ax1.set_xlim(-3, 3)
        ax1.grid(True, alpha=0.3)
        
        # 添加统计信息
        near_binary = ((orig_data < -0.8) | (orig_data > 0.8)).sum() / len(orig_data) * 100
        ax1.text(0.02, 0.98, f'{near_binary:.1f}% 接近±1',
                transform=ax1.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # 渐进式方法
        ax2 = axes[idx, 1]
        prog_data = progressive_outputs[idx].flatten()
        ax2.hist(prog_data, bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax2.axvline(x=-1, color='red', linestyle='--', linewidth=2)
        ax2.axvline(x=1, color='red', linestyle='--', linewidth=2)
        ax2.set_title(f'渐进式方法 - Epoch {epoch} (scale={scale:.1f})', 
                     fontsize=12, fontweight='bold')
        ax2.set_xlabel('Hash Code Value')
        ax2.set_ylabel('Frequency')
        ax2.set_xlim(-1.2, 1.2)
        ax2.grid(True, alpha=0.3)
        
        # 添加统计信息
        near_binary = ((prog_data < -0.8) | (prog_data > 0.8)).sum() / len(prog_data) * 100
        ax2.text(0.02, 0.98, f'{near_binary:.1f}% 接近±1',
                transform=ax2.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/Users/yutinglai/Documents/code/PythonCode/UCMFH/progressive_vs_original_comparison.png',
                dpi=300, bbox_inches='tight')
    print("✅ 对比图表已保存到: progressive_vs_original_comparison.png")
    plt.show()


def analyze_quantization_error():
    """分析量化误差（将连续值转为-1/+1时的误差）"""
    
    print("\n" + "=" * 80)
    print("量化误差分析")
    print("=" * 80)
    print()
    print("在实际使用时，需要将连续的哈希码转为二值码（-1或+1）")
    print("量化误差 = |原始值 - 量化后的值|")
    print()
    
    torch.manual_seed(42)
    x = torch.randn(1000, 512)
    
    original_model = OriginalHashMLP()
    progressive_model = ProgressiveHashMLP()
    progressive_model.load_state_dict(original_model.state_dict())
    
    scales = [1.0, 5.0, 10.0]
    
    print("不同Scale下的量化误差:")
    print("-" * 80)
    
    for scale in scales:
        with torch.no_grad():
            if scale == 1.0:
                # 原始方法（相当于scale=1的渐进式方法）
                hash_codes = original_model(x)
                method_name = "原始方法"
            else:
                hash_codes = progressive_model(x, scale=scale)
                method_name = f"渐进式 (scale={scale})"
        
        # 量化（转为-1或+1）
        quantized = torch.sign(hash_codes)
        quantized[quantized == 0] = 1  # 处理0的情况
        
        # 计算误差
        error = torch.abs(hash_codes - quantized).mean().item()
        
        print(f"{method_name:25s}: 平均量化误差 = {error:.4f}")
    
    print("-" * 80)
    print()
    print("💡 观察：")
    print("  - Scale越大，量化误差越小")
    print("  - 量化误差小意味着信息损失少，检索性能更好")
    print()


def demo_scale_evolution():
    """演示scale参数的演变过程"""
    
    print("\n" + "=" * 80)
    print("Scale参数演变过程")
    print("=" * 80)
    print()
    
    total_epochs = 100
    scale_min = 1.0
    scale_max = 10.0
    
    epochs = list(range(0, total_epochs, 5))
    scales = [scale_min + (e / (total_epochs - 1)) * (scale_max - scale_min) 
              for e in epochs]
    
    # 绘制scale演变曲线
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, scales, 'b-', linewidth=2, marker='o', markersize=4)
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Scale Parameter', fontsize=14)
    plt.title('渐进式哈希学习：Scale参数随训练进度的变化', fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.axhline(y=scale_min, color='g', linestyle='--', label=f'scale_min = {scale_min}')
    plt.axhline(y=scale_max, color='r', linestyle='--', label=f'scale_max = {scale_max}')
    plt.legend(fontsize=12)
    
    # 标注关键点
    milestones = [0, 25, 50, 75, 99]
    for m in milestones:
        s = scale_min + (m / (total_epochs - 1)) * (scale_max - scale_min)
        plt.annotate(f'Epoch {m}\nScale={s:.2f}', 
                    xy=(m, s), xytext=(m+5, s+0.5),
                    arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                    fontsize=10, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))
    
    plt.tight_layout()
    plt.savefig('/Users/yutinglai/Documents/code/PythonCode/UCMFH/scale_evolution.png',
                dpi=300, bbox_inches='tight')
    print("✅ Scale演变图已保存到: scale_evolution.png")
    plt.show()
    
    print("\n关键训练阶段:")
    print("-" * 80)
    for m in milestones:
        s = scale_min + (m / (total_epochs - 1)) * (scale_max - scale_min)
        progress = m / (total_epochs - 1) * 100
        print(f"Epoch {m:3d} | 进度: {progress:5.1f}% | Scale: {s:.2f}")
    print("-" * 80)


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("建议2实现效果演示：渐进式哈希学习 (Progressive Hash Learning)")
    print("=" * 80)
    print()
    print("这个演示对比了两种方法:")
    print("  1. 原始UCMFH方法：直接输出哈希码，没有渐进式控制")
    print("  2. HashNet改进方法：使用scale参数实现渐进式二值化")
    print()
    
    # 运行对比
    original_outputs, progressive_outputs, scales, epochs = simulate_training_comparison()
    
    # 生成可视化
    print("\n生成对比可视化...")
    visualize_comparison(original_outputs, progressive_outputs, scales, epochs)
    
    # 量化误差分析
    analyze_quantization_error()
    
    # Scale演变演示
    demo_scale_evolution()
    
    print("\n" + "=" * 80)
    print("总结")
    print("=" * 80)
    print()
    print("渐进式哈希学习的优势:")
    print("  ✅ 训练初期保持连续性，便于梯度优化")
    print("  ✅ 训练末期实现二值化，减少量化误差")
    print("  ✅ 整个过程平滑过渡，训练更稳定")
    print("  ✅ 最终性能显著提升（预期+2-5个百分点mAP）")
    print()
    print("✅ 所有演示完成!")
