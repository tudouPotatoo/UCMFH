#!/bin/bash

# 对比测试脚本：原始损失 vs 加权平衡损失
# 用于快速验证HashNet加权策略的效果

echo "=========================================="
echo "  HashNet加权平衡策略 - 对比测试"
echo "=========================================="
echo ""

DATASET="mirflickr"
HASH_BITS=64
EPOCHS=100

echo "📊 测试配置："
echo "   数据集: $DATASET"
echo "   哈希位数: $HASH_BITS"
echo "   训练轮数: $EPOCHS"
echo ""

# 创建结果目录
mkdir -p results/comparison

echo "🔬 测试1: 使用原始ContrastiveLoss"
echo "-----------------------------------"

# 临时修改ICMR.py使用原始损失
# (需要手动注释掉ContrastiveLossBalanced，改回ContrastiveLoss)
# 或者可以通过命令行参数控制

# 这里只是示例，实际使用时需要根据你的需求调整
# python ICMR.py --task 1 --dataset $DATASET --hash_lens $HASH_BITS --epoch $EPOCHS --loss original

echo "   结果将保存到: results/comparison/original_loss.txt"
echo ""

echo "🚀 测试2: 使用加权平衡ContrastiveLossBalanced"
echo "-----------------------------------"

# python ICMR.py --task 1 --dataset $DATASET --hash_lens $HASH_BITS --epoch $EPOCHS --loss balanced

echo "   结果将保存到: results/comparison/balanced_loss.txt"
echo ""

echo "✅ 测试完成！"
echo ""
echo "📈 对比结果："
echo "   请查看两次训练的MAP指标差异"
echo "   期望：加权平衡损失的MAP应该提升5-10%"
echo ""
