"""
测试加权平衡损失函数

这个脚本演示了ContrastiveLossBalanced相比原始ContrastiveLoss的改进。
可以通过简单的示例看到加权平衡的效果。
"""

import torch
import torch.nn.functional as F
from metric import ContrastiveLoss, ContrastiveLossBalanced

def test_loss_comparison():
    """对比原始损失和加权平衡损失"""
    
    print("=" * 70)
    print("对比原始ContrastiveLoss和加权平衡的ContrastiveLossBalanced")
    print("=" * 70)
    
    # 设置参数
    batch_size = 8
    dim = 512
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 创建两个损失函数
    loss_original = ContrastiveLoss(batch_size=batch_size, device=device)
    loss_balanced = ContrastiveLossBalanced(batch_size=batch_size, device=device)
    
    # 创建模拟数据
    # 模拟场景：图像和文本嵌入
    torch.manual_seed(42)
    emb_i = torch.randn(batch_size, dim).to(device)  # 图像嵌入
    emb_j = torch.randn(batch_size, dim).to(device)  # 文本嵌入
    
    # 让配对的图文更相似（模拟真实情况）
    for i in range(batch_size):
        # 让第i个图像和第i个文本更相似
        emb_j[i] = emb_j[i] + 0.5 * emb_i[i]
    
    # 计算两种损失
    with torch.no_grad():
        loss1 = loss_original(emb_i, emb_j)
        loss2 = loss_balanced(emb_i, emb_j)
    
    print(f"\n📊 损失值对比：")
    print(f"原始 ContrastiveLoss:        {loss1.item():.4f}")
    print(f"加权 ContrastiveLossBalanced: {loss2.item():.4f}")
    print(f"差异:                        {abs(loss1.item() - loss2.item()):.4f}")
    
    # 分析权重差异
    print(f"\n💡 加权机制分析：")
    print(f"Batch size: {batch_size}")
    print(f"正样本对数量: {batch_size} (对角线，配对的图文)")
    print(f"负样本对数量: {batch_size * (batch_size - 1)} (非对角线，不配对的图文)")
    
    S1 = batch_size
    S0 = batch_size * (batch_size - 1)
    S = S1 + S0
    
    positive_weight = S / S1
    negative_weight = S / S0
    
    print(f"\n⚖️  权重比例：")
    print(f"正样本权重: {positive_weight:.2f}x")
    print(f"负样本权重: {negative_weight:.2f}x")
    print(f"权重比: {positive_weight / negative_weight:.2f}:1")
    
    print(f"\n✅ 这意味着模型会更加关注'什么是相似的'（正样本）")
    print(f"   而不是简单记住'大部分都不相似'（负样本）")
    
    return loss1, loss2


def visualize_weight_effect():
    """可视化不同batch size下的权重效果"""
    
    print("\n" + "=" * 70)
    print("不同Batch Size下的权重效果")
    print("=" * 70)
    
    batch_sizes = [4, 8, 16, 32, 64, 128]
    
    print(f"\n{'Batch Size':<12} {'正样本数':<10} {'负样本数':<12} {'正权重':<10} {'负权重':<10} {'权重比':<10}")
    print("-" * 70)
    
    for bs in batch_sizes:
        S1 = bs
        S0 = bs * (bs - 1)
        S = S1 + S0
        
        pos_weight = S / S1
        neg_weight = S / S0
        weight_ratio = pos_weight / neg_weight
        
        print(f"{bs:<12} {S1:<10} {S0:<12} {pos_weight:<10.2f} {neg_weight:<10.2f} {weight_ratio:<10.2f}")
    
    print("\n💡 观察：")
    print("   - Batch size越大，正负样本的不平衡越严重")
    print("   - 权重比也随之增大，更强调正样本的重要性")
    print("   - 这正是加权平衡策略的价值所在！")


def test_gradient_comparison():
    """对比两种损失函数的梯度"""
    
    print("\n" + "=" * 70)
    print("梯度对比测试（检查哪种损失函数学习效率更高）")
    print("=" * 70)
    
    batch_size = 8
    dim = 512
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    
    # 创建可训练的嵌入
    emb_i = torch.randn(batch_size, dim, requires_grad=True, device=device)
    emb_j = torch.randn(batch_size, dim, requires_grad=True, device=device)
    
    # 测试原始损失
    loss_fn1 = ContrastiveLoss(batch_size=batch_size, device=device)
    loss1 = loss_fn1(emb_i, emb_j)
    loss1.backward()
    grad_norm1 = emb_i.grad.norm().item()
    
    # 重置梯度
    emb_i.grad = None
    emb_j.grad = None
    
    # 测试加权损失
    loss_fn2 = ContrastiveLossBalanced(batch_size=batch_size, device=device)
    loss2 = loss_fn2(emb_i, emb_j)
    loss2.backward()
    grad_norm2 = emb_i.grad.norm().item()
    
    print(f"\n📈 梯度范数对比：")
    print(f"原始损失梯度范数: {grad_norm1:.4f}")
    print(f"加权损失梯度范数: {grad_norm2:.4f}")
    print(f"差异比例: {grad_norm2/grad_norm1:.2f}x")
    
    if grad_norm2 > grad_norm1:
        print(f"\n✅ 加权损失产生了更大的梯度，这通常意味着：")
        print(f"   - 对正样本的学习信号更强")
        print(f"   - 模型会更快地学习'什么是相似的'")
    

if __name__ == "__main__":
    print("\n" + "🔬" * 35)
    print("HashNet加权平衡策略 - 测试脚本")
    print("🔬" * 35)
    
    # 测试1：基本损失对比
    test_loss_comparison()
    
    # 测试2：可视化权重效果
    visualize_weight_effect()
    
    # 测试3：梯度对比
    test_gradient_comparison()
    
    print("\n" + "=" * 70)
    print("✅ 测试完成！")
    print("=" * 70)
    print("\n💡 总结：")
    print("   1. ContrastiveLossBalanced通过加权平衡正负样本")
    print("   2. 正样本权重 >> 负样本权重，强制模型关注相似性学习")
    print("   3. 这是HashNet的核心思想，能有效缓解类别不平衡问题")
    print("\n🚀 现在可以开始训练了！期待性能提升！\n")
