"""
可视化对比：原始损失 vs 加权平衡损失

这个脚本创建简单的可视化图表，帮助理解加权平衡策略的工作原理
不需要训练数据，纯粹展示理论效果
"""

def print_ascii_comparison():
    """用ASCII艺术展示权重对比"""
    
    print("\n" + "="*70)
    print("加权平衡策略 - 可视化对比")
    print("="*70)
    
    batch_size = 32
    positive_samples = batch_size
    negative_samples = batch_size * (batch_size - 1)
    
    print(f"\n📊 场景：Batch Size = {batch_size}")
    print(f"   正样本（配对图文）：{positive_samples} 个")
    print(f"   负样本（不配对）  ：{negative_samples} 个")
    print(f"   不平衡比例        ：1:{negative_samples//positive_samples}")
    
    print("\n" + "-"*70)
    print("原始ContrastiveLoss（无加权）")
    print("-"*70)
    
    # 正样本重要性
    pos_importance_original = 32
    neg_importance_original = 992
    
    print("\n正样本重要性：")
    print("▓" * (pos_importance_original // 4))
    print(f"({pos_importance_original} 单位)")
    
    print("\n负样本重要性：")
    print("▓" * 50 + "... (太长了，无法完全显示)")
    print(f"({neg_importance_original} 单位)")
    
    print("\n❌ 问题：负样本占据主导地位，模型容易只学会'都不相似'")
    
    print("\n" + "-"*70)
    print("加权平衡ContrastiveLossBalanced（HashNet策略）")
    print("-"*70)
    
    # 计算权重
    total = positive_samples + negative_samples
    pos_weight = total / positive_samples
    neg_weight = total / negative_samples
    
    # 加权后的重要性
    pos_importance_balanced = pos_importance_original * pos_weight
    neg_importance_balanced = neg_importance_original * neg_weight
    
    print("\n正样本重要性（加权后）：")
    bars = int(pos_importance_balanced / 20)
    print("▓" * bars)
    print(f"({pos_importance_balanced:.0f} 单位，权重={pos_weight:.1f}x)")
    
    print("\n负样本重要性（加权后）：")
    bars = int(neg_importance_balanced / 20)
    print("▓" * bars)
    print(f"({neg_importance_balanced:.0f} 单位，权重={neg_weight:.2f}x)")
    
    print("\n✅ 效果：正负样本达到平衡，模型被迫学习'什么是相似的'")
    
    print("\n" + "="*70)
    print("权重比例详细分析")
    print("="*70)
    
    print(f"\n正样本权重提升：{pos_weight:.1f}倍")
    print(f"负样本权重调整：{neg_weight:.3f}倍（接近1，几乎不变）")
    print(f"正负权重比    ：{pos_weight/neg_weight:.1f}:1")
    
    print("\n💡 这个比例正好等于原始的不平衡比例！")
    print(f"   原始不平衡：1:{negative_samples//positive_samples}")
    print(f"   权重补偿  ：{pos_weight/neg_weight:.0f}:1")
    print(f"   最终效果  ：平衡 ✅")


def print_batch_size_impact():
    """展示不同batch size的影响"""
    
    print("\n" + "="*70)
    print("不同Batch Size下的加权效果")
    print("="*70)
    
    print(f"\n{'Batch':<8} {'正样本':<8} {'负样本':<10} {'正权重':<10} {'负权重':<10} {'不平衡':<12} {'权重比':<10}")
    print("-"*70)
    
    for bs in [8, 16, 32, 64, 128, 256]:
        pos = bs
        neg = bs * (bs - 1)
        total = pos + neg
        
        pos_w = total / pos
        neg_w = total / neg
        
        imbalance = f"1:{neg//pos}"
        weight_ratio = f"{pos_w/neg_w:.0f}:1"
        
        print(f"{bs:<8} {pos:<8} {neg:<10} {pos_w:<10.1f} {neg_w:<10.3f} {imbalance:<12} {weight_ratio:<10}")
    
    print("\n📈 观察：")
    print("   • Batch size越大，不平衡越严重")
    print("   • 加权策略自动适应，权重比=不平衡比")
    print("   • 建议使用batch_size≥32以获得最佳效果")


def print_training_tips():
    """打印训练建议"""
    
    print("\n" + "="*70)
    print("🚀 训练建议")
    print("="*70)
    
    tips = [
        ("增大Batch Size", "从128增加到256或更大", "加权效果更明显"),
        ("调整温度参数", "尝试temperature=0.3或0.7", "控制模型挑剔程度"),
        ("监控损失趋势", "损失值会变大但要持续下降", "看趋势不看绝对值"),
        ("对比MAP指标", "保存两组模型分别测试", "量化改进效果"),
        ("耐心训练", "可能需要更多epoch才能收敛", "但最终效果更好"),
    ]
    
    print()
    for i, (action, how, why) in enumerate(tips, 1):
        print(f"{i}. {action}")
        print(f"   如何做：{how}")
        print(f"   为什么：{why}")
        print()


if __name__ == "__main__":
    print("\n" + "🎨"*35)
    print("HashNet加权平衡策略 - 可视化说明")
    print("🎨"*35)
    
    # 1. ASCII对比图
    print_ascii_comparison()
    
    # 2. Batch size影响分析
    print_batch_size_impact()
    
    # 3. 训练建议
    print_training_tips()
    
    print("\n" + "="*70)
    print("✅ 可视化完成！")
    print("="*70)
    print("\n现在你应该明白了：")
    print("   1️⃣  为什么需要加权：正负样本严重不平衡")
    print("   2️⃣  如何加权：给正样本更大权重")
    print("   3️⃣  效果如何：强制模型学习相似性，而非只记住不相似")
    print("\n开始训练吧！期待你的好结果！🎉\n")
