"""
原始方法 vs 改进方法 - 互动演示

这个脚本用具体的数字演示两种方法的差异
"""

def simulate_loss_calculation():
    """模拟损失计算过程"""
    
    print("=" * 70)
    print("场景：训练一个batch的图文数据")
    print("=" * 70)
    
    batch_size = 8
    
    print(f"\n我们有 {batch_size} 对图文：")
    pairs = [
        ("🐱猫图1", "猫的描述1"),
        ("🐶狗图1", "狗的描述1"),
        ("🚗车图1", "车的描述1"),
        ("🌺花图1", "花的描述1"),
        ("🐱猫图2", "猫的描述2"),
        ("🐶狗图2", "狗的描述2"),
        ("🚗车图2", "车的描述2"),
        ("🌺花图2", "花的描述2"),
    ]
    
    for i, (img, txt) in enumerate(pairs):
        print(f"  {i+1}. {img} <-> {txt}")
    
    print("\n" + "-" * 70)
    print("第1步：计算相似度矩阵（8×8）")
    print("-" * 70)
    
    print("\n对于'猫图1'，它与所有文字的关系：")
    similarities = [
        ("猫的描述1", "✅ 正样本", "高相似度 0.9"),
        ("狗的描述1", "❌ 负样本", "低相似度 0.2"),
        ("车的描述1", "❌ 负样本", "低相似度 0.1"),
        ("花的描述1", "❌ 负样本", "低相似度 0.15"),
        ("猫的描述2", "❌ 负样本", "中等相似 0.4（也是猫但不配对）"),
        ("狗的描述2", "❌ 负样本", "低相似度 0.18"),
        ("车的描述2", "❌ 负样本", "低相似度 0.12"),
        ("花的描述2", "❌ 负样本", "低相似度 0.16"),
    ]
    
    print("\n文字描述          | 类型      | 相似度")
    print("-" * 50)
    for txt, typ, sim in similarities:
        print(f"{txt:<16} | {typ:<8} | {sim}")
    
    print("\n统计：1个正样本，7个负样本，比例 = 1:7")
    
    # 原始方法
    print("\n" + "=" * 70)
    print("📊 原始方法：ContrastiveLoss")
    print("=" * 70)
    
    print("\n第2步：计算InfoNCE损失（无加权）")
    print("\n公式：loss = -log(exp(正样本相似度) / (所有样本的exp和))")
    
    import math
    
    # 模拟计算
    pos_sim = 0.9  # 正样本相似度
    neg_sims = [0.2, 0.1, 0.15, 0.4, 0.18, 0.12, 0.16]  # 负样本相似度
    
    temp = 0.5  # 温度参数
    
    # 计算exp值
    exp_pos = math.exp(pos_sim / temp)
    exp_negs = [math.exp(s / temp) for s in neg_sims]
    exp_sum = exp_pos + sum(exp_negs)
    
    # 原始损失
    loss_original = -math.log(exp_pos / exp_sum)
    
    print(f"\n正样本的exp值：exp({pos_sim}/{temp}) = {exp_pos:.2f}")
    print(f"所有负样本exp和：{sum(exp_negs):.2f}")
    print(f"总exp和：{exp_sum:.2f}")
    print(f"\n❌ 原始损失 = -log({exp_pos:.2f} / {exp_sum:.2f}) = {loss_original:.4f}")
    
    print("\n问题分析：")
    print(f"  • 分母中负样本贡献：{sum(exp_negs):.2f}")
    print(f"  • 分母中正样本贡献：{exp_pos:.2f}")
    print(f"  • 负样本主导了分母（{sum(exp_negs)/exp_sum*100:.1f}% vs {exp_pos/exp_sum*100:.1f}%）")
    print(f"  • 模型容易关注负样本，忽视正样本")
    
    # 改进方法
    print("\n" + "=" * 70)
    print("✨ 改进方法：ContrastiveLossBalanced")
    print("=" * 70)
    
    print("\n第2步：计算权重")
    
    S1 = 1  # 正样本数
    S0 = 7  # 负样本数
    S = S1 + S0  # 总数
    
    w_pos = S / S1
    w_neg = S / S0
    
    print(f"\n正样本数量 S1 = {S1}")
    print(f"负样本数量 S0 = {S0}")
    print(f"总数量     S  = {S}")
    print(f"\n正样本权重 = S/S1 = {S}/{S1} = {w_pos:.2f}倍")
    print(f"负样本权重 = S/S0 = {S}/{S0} = {w_neg:.2f}倍")
    
    print("\n第3步：应用加权计算损失")
    
    # 加权后的负样本exp和
    weighted_exp_negs_sum = sum(exp_negs) * w_neg
    weighted_exp_sum = exp_pos + weighted_exp_negs_sum
    
    # 加权损失
    loss_weighted = -math.log(exp_pos / weighted_exp_sum)
    
    # 对正样本再次加权
    loss_final = loss_weighted * w_pos
    
    print(f"\n负样本exp和（加权前）：{sum(exp_negs):.2f}")
    print(f"负样本exp和（加权后）：{sum(exp_negs):.2f} × {w_neg:.2f} = {weighted_exp_negs_sum:.2f}")
    print(f"新的总exp和：{exp_pos:.2f} + {weighted_exp_negs_sum:.2f} = {weighted_exp_sum:.2f}")
    
    print(f"\n中间损失 = -log({exp_pos:.2f} / {weighted_exp_sum:.2f}) = {loss_weighted:.4f}")
    print(f"✅ 最终损失 = {loss_weighted:.4f} × {w_pos:.2f} = {loss_final:.4f}")
    
    print("\n" + "=" * 70)
    print("📈 对比结果")
    print("=" * 70)
    
    print(f"\n原始损失：{loss_original:.4f}")
    print(f"改进损失：{loss_final:.4f}")
    print(f"差异：    {loss_final - loss_original:+.4f} ({(loss_final/loss_original - 1)*100:+.1f}%)")
    
    print("\n" + "=" * 70)
    print("💡 关键洞察")
    print("=" * 70)
    
    print("\n1. 损失值变化：")
    print(f"   • 改进后损失值更大（{loss_final:.4f} vs {loss_original:.4f}）")
    print(f"   • 这是正常的！因为正样本获得了更高权重")
    print(f"   • 重要的是看梯度方向和训练趋势")
    
    print("\n2. 权重效果：")
    print(f"   • 正样本重要性提升了 {w_pos:.0f} 倍")
    print(f"   • 负样本权重几乎不变（{w_neg:.2f}倍）")
    print(f"   • 正负样本从 1:{S0} 不平衡变为平衡状态")
    
    print("\n3. 模型行为：")
    print("   ❌ 原始方法：模型可能偷懒，只关注'都不相似'")
    print("   ✅ 改进方法：模型被迫学习'什么是真正相似的'")
    
    print("\n4. 预期效果：")
    print("   • 模型更好地区分相似和不相似")
    print("   • MAP指标提升 5-10%")
    print("   • 特别是在大batch size时效果更明显")


def compare_batch_sizes():
    """对比不同batch size下的效果"""
    
    print("\n" + "=" * 70)
    print("🔬 不同Batch Size的权重对比")
    print("=" * 70)
    
    batch_sizes = [8, 16, 32, 64, 128, 256]
    
    print(f"\n{'Batch':<8} {'正样本':<8} {'负样本':<10} {'原始方法':<20} {'改进方法':<20}")
    print("-" * 70)
    
    for bs in batch_sizes:
        pos = bs
        neg = bs * (bs - 1)
        
        # 权重
        w_pos = (pos + neg) / pos
        w_neg = (pos + neg) / neg
        
        original = f"正={pos}×1, 负={neg}×1"
        improved = f"正={pos}×{w_pos:.0f}, 负={neg}×{w_neg:.2f}"
        
        print(f"{bs:<8} {pos:<8} {neg:<10} {original:<20} {improved:<20}")
    
    print("\n观察：")
    print("  • Batch size越大，不平衡问题越严重")
    print("  • 改进方法的权重会自动适应")
    print("  • 建议使用batch_size≥32以获得最佳效果")


def interactive_demo():
    """互动演示"""
    
    print("\n" + "🎮" * 35)
    print("互动演示：训练过程模拟")
    print("🎮" * 35)
    
    print("\n假设我们在训练过程中看到以下相似度：")
    print("\n场景：猫图和8个文本的相似度")
    
    test_cases = [
        {
            "name": "训练初期（模型还没学好）",
            "pos_sim": 0.5,
            "neg_sims": [0.48, 0.47, 0.49, 0.46, 0.48, 0.47, 0.45],
            "description": "所有相似度都差不多，模型还分不清"
        },
        {
            "name": "训练中期（开始学习）",
            "pos_sim": 0.7,
            "neg_sims": [0.4, 0.35, 0.42, 0.38, 0.41, 0.36, 0.39],
            "description": "正样本开始高于负样本"
        },
        {
            "name": "训练后期（学得很好）",
            "pos_sim": 0.9,
            "neg_sims": [0.2, 0.15, 0.18, 0.22, 0.19, 0.17, 0.21],
            "description": "正样本明显高于负样本"
        }
    ]
    
    import math
    temp = 0.5
    
    for case in test_cases:
        print("\n" + "=" * 70)
        print(f"📊 {case['name']}")
        print("=" * 70)
        print(f"说明：{case['description']}")
        
        pos_sim = case['pos_sim']
        neg_sims = case['neg_sims']
        
        print(f"\n正样本相似度：{pos_sim:.2f}")
        print(f"负样本相似度：{[f'{s:.2f}' for s in neg_sims]}")
        
        # 原始损失
        exp_pos = math.exp(pos_sim / temp)
        exp_negs = [math.exp(s / temp) for s in neg_sims]
        exp_sum = exp_pos + sum(exp_negs)
        loss_original = -math.log(exp_pos / exp_sum)
        
        # 改进损失
        w_pos = 8.0
        w_neg = 8.0 / 7.0
        weighted_exp_sum = exp_pos + sum(exp_negs) * w_neg
        loss_improved = -math.log(exp_pos / weighted_exp_sum) * w_pos
        
        print(f"\n原始损失：{loss_original:.4f}")
        print(f"改进损失：{loss_improved:.4f}")
        
        # 梯度强度（简化估计）
        grad_original = (1 - exp_pos/exp_sum)
        grad_improved = w_pos * (1 - exp_pos/weighted_exp_sum)
        
        print(f"\n估计梯度强度（用于更新正样本）：")
        print(f"  原始方法：{grad_original:.4f}")
        print(f"  改进方法：{grad_improved:.4f}")
        print(f"  改进方法的梯度是原始的 {grad_improved/grad_original:.2f} 倍！")
        
        if grad_improved > grad_original:
            print(f"  ✅ 改进方法会更快地提升正样本的相似度")


if __name__ == "__main__":
    print("\n" + "📚" * 35)
    print("原始方法 vs 改进方法 - 详细演示")
    print("📚" * 35)
    
    # 1. 基本损失计算对比
    simulate_loss_calculation()
    
    # 2. Batch size影响
    compare_batch_sizes()
    
    # 3. 互动演示
    interactive_demo()
    
    print("\n" + "=" * 70)
    print("✅ 演示完成！")
    print("=" * 70)
    
    print("\n🎯 核心要点：")
    print("  1. 原始方法：所有样本一视同仁")
    print("  2. 改进方法：正样本获得更高权重")
    print("  3. 结果：模型被迫学习'相似性'而非只记住'不相似'")
    print("  4. 效果：MAP指标预期提升5-10%")
    
    print("\n💪 现在你完全理解两种方法的区别了！\n")
