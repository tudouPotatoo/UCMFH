"""
UCMFH with HashNet Weighted Balance Strategy - Quick Start Example
快速开始示例
"""

# 示例1: 在mirflickr数据集上使用加权平衡策略训练16位哈希码
# Example 1: Train 16-bit hash codes with weighted balance strategy on mirflickr
# python demo.py --dataset mirflickr --hash_lens 16 --epoch 50 --task 1 --use_weighted_balance

# 示例2: 在mscoco数据集上使用加权平衡策略训练32位哈希码  
# Example 2: Train 32-bit hash codes with weighted balance strategy on mscoco
# python demo.py --dataset mscoco --hash_lens 32 --epoch 50 --task 1 --use_weighted_balance

# 示例3: 对比实验 - 不使用加权平衡策略
# Example 3: Comparison - without weighted balance strategy
# python demo.py --dataset mirflickr --hash_lens 16 --epoch 50 --task 1

# 示例4: 在nus-wide数据集上使用加权平衡策略训练64位哈希码
# Example 4: Train 64-bit hash codes with weighted balance strategy on nus-wide
# python demo.py --dataset nus-wide --hash_lens 64 --epoch 50 --task 1 --use_weighted_balance

# 示例5: 测试已训练的模型
# Example 5: Test trained model
# python demo.py --dataset mirflickr --hash_lens 16 --task 3

"""
代码修改总结 / Code Modification Summary:

1. metric.py:
   - 新增 WeightedBalanceLoss 类
   - 实现HashNet的加权平衡策略
   
2. ICMR.py:
   - 导入 WeightedBalanceLoss
   - 添加 use_weighted_balance 配置参数
   - 在 trainhash() 中根据配置选择使用对应的loss函数
   
3. demo.py:
   - 添加 --use_weighted_balance 命令行参数

4. 新增文件:
   - test-weighted-balance.sh: 批量测试脚本
   - README_WEIGHTED_BALANCE.md: 详细说明文档
   - USAGE_EXAMPLES.py: 使用示例（本文件）
"""

# 如果要在Python代码中直接调用，可以这样做：
if __name__ == "__main__":
    import os
    import sys
    
    # 设置CUDA设备
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    
    # 构造命令行参数
    # 使用加权平衡策略
    cmd_with_balance = "python demo.py --dataset mirflickr --hash_lens 16 --epoch 50 --task 1 --use_weighted_balance"
    
    # 不使用加权平衡策略（对比实验）
    cmd_without_balance = "python demo.py --dataset mirflickr --hash_lens 16 --epoch 50 --task 1"
    
    print("使用加权平衡策略训练:")
    print(cmd_with_balance)
    print("\n不使用加权平衡策略训练(对比):")
    print(cmd_without_balance)
    
    # 取消下面的注释来实际运行
    # os.system(cmd_with_balance)
