"""
测试渐进式哈希学习的效果

这个脚本演示了随着训练进度增加，scale参数如何影响哈希码的分布
"""
import torch
import numpy as np
import matplotlib.pyplot as plt

def visualize_progressive_hash():
    """可视化渐进式哈希学习过程"""
    
    # 模拟一批哈希码原始值（训练过程中的连续值）
    torch.manual_seed(42)
    x = torch.randn(100) * 0.5  # 100个样本，范围大致在[-1, 1]之间
    
    # 不同训练阶段的scale值
    scales = [1.0, 2.0, 4.0, 6.0, 8.0, 10.0]
    epochs = [0, 20, 40, 60, 80, 100]  # 假设总共100个epoch
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()
    
    for idx, (scale, epoch) in enumerate(zip(scales, epochs)):
        # 应用渐进式变换
        y = torch.tanh(scale * x)
        
        # 绘制直方图
        ax = axes[idx]
        ax.hist(y.numpy(), bins=50, alpha=0.7, color='blue', edgecolor='black')
        ax.axvline(x=-1, color='red', linestyle='--', linewidth=2, label='目标值 -1')
        ax.axvline(x=1, color='red', linestyle='--', linewidth=2, label='目标值 +1')
        ax.set_title(f'Epoch {epoch} (scale={scale})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Hash Code Value', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_xlim(-1.2, 1.2)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 添加统计信息
        close_to_binary = ((y < -0.8) | (y > 0.8)).sum().item() / len(y) * 100
        ax.text(0.02, 0.98, f'{close_to_binary:.1f}% 接近二值化',
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig('/Users/yutinglai/Documents/code/PythonCode/UCMFH/progressive_hash_visualization.png', 
                dpi=300, bbox_inches='tight')
    print("✅ 可视化图表已保存到: progressive_hash_visualization.png")
    plt.show()


def compare_hash_distributions():
    """对比不同scale下的哈希码分布"""
    
    torch.manual_seed(42)
    x = torch.randn(1000) * 0.5
    
    scales = [1.0, 5.0, 10.0]
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    for scale in scales:
        y = torch.tanh(scale * x)
        ax.hist(y.numpy(), bins=50, alpha=0.5, label=f'scale={scale}', edgecolor='black')
    
    ax.axvline(x=-1, color='red', linestyle='--', linewidth=2)
    ax.axvline(x=1, color='red', linestyle='--', linewidth=2)
    ax.set_xlabel('Hash Code Value', fontsize=14)
    ax.set_ylabel('Frequency', fontsize=14)
    ax.set_title('不同Scale参数下的哈希码分布对比', fontsize=16, fontweight='bold')
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('/Users/yutinglai/Documents/code/PythonCode/UCMFH/scale_comparison.png', 
                dpi=300, bbox_inches='tight')
    print("✅ 对比图表已保存到: scale_comparison.png")
    plt.show()


def simulate_training_progress():
    """模拟完整的训练过程"""
    
    print("=" * 70)
    print("模拟渐进式哈希学习训练过程")
    print("=" * 70)
    
    total_epochs = 100
    scale_min = 1.0
    scale_max = 10.0
    
    # 模拟数据
    torch.manual_seed(42)
    x = torch.randn(1000) * 0.5
    
    print(f"\n训练配置:")
    print(f"  总epoch数: {total_epochs}")
    print(f"  Scale范围: {scale_min} → {scale_max}")
    print(f"  样本数量: {len(x)}")
    print()
    
    # 记录训练过程中的统计信息
    epochs_to_check = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]
    
    print("训练进度 | Scale | 接近二值化比例 | 平均绝对值")
    print("-" * 70)
    
    for epoch in epochs_to_check:
        # 计算当前scale
        progress = epoch / max(total_epochs - 1, 1)
        current_scale = scale_min + progress * (scale_max - scale_min)
        
        # 应用变换
        y = torch.tanh(current_scale * x)
        
        # 统计信息
        close_to_binary = ((y < -0.8) | (y > 0.8)).sum().item() / len(y) * 100
        mean_abs = torch.abs(y).mean().item()
        
        print(f"  {epoch:3d}/{total_epochs}  | {current_scale:5.2f} |    {close_to_binary:5.1f}%     |   {mean_abs:.3f}")
    
    print("=" * 70)
    print("\n💡 观察：")
    print("  - 训练初期（scale≈1）：哈希码分布较分散，便于优化")
    print("  - 训练中期（scale≈5）：哈希码逐渐向±1靠拢")
    print("  - 训练末期（scale≈10）：大部分哈希码接近±1，实现二值化")
    print()


if __name__ == "__main__":
    print("\n" + "=" * 70)
    print("渐进式哈希学习 (Progressive Hash Learning) 测试")
    print("=" * 70)
    print()
    print("这个测试演示了HashNet的核心思想：")
    print("  1. 训练初期使用小的scale值，让哈希码保持连续性")
    print("  2. 训练过程中逐渐增大scale，推动哈希码向±1靠拢")
    print("  3. 训练末期使用大的scale值，实现接近二值化的效果")
    print()
    
    # 运行测试
    simulate_training_progress()
    print("\n生成可视化图表...")
    visualize_progressive_hash()
    compare_hash_distributions()
    
    print("\n✅ 所有测试完成!")
