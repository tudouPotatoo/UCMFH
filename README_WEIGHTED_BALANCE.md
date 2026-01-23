# UCMFH with HashNet Weighted Balance Strategy

## 修改说明

本次修改在UCMFH代码基础上集成了HashNet的加权平衡策略（Weighted Balance Strategy），用于处理跨模态哈希学习中的类别不均衡问题。

## 主要改动

### 1. 新增加权平衡Loss函数 (`metric.py`)

新增了 `WeightedBalanceLoss` 类，实现了HashNet的加权平衡策略：

```python
class WeightedBalanceLoss(nn.Module):
    """
    HashNet's weighted balance strategy for handling class imbalance.
    Based on pairwise_loss_updated from HashNet.
    """
```

**核心思想**:
- 计算正样本对（相似对）和负样本对（不相似对）的数量
- 对数量较少的类别赋予更高的权重
- 平衡策略: `weight = S / S_class`，其中S是总样本对数，S_class是该类别的样本对数

### 2. 集成到训练流程 (`ICMR.py`)

修改了 `Solver` 类：
- 初始化时添加 `use_weighted_balance` 参数控制
- 在 `trainhash()` 函数中，根据配置选择使用加权平衡Loss或原始ContrastiveLoss
- 传入label信息用于计算样本相似度

### 3. 添加命令行参数 (`demo.py`)

新增参数:
```bash
--use_weighted_balance    # 使用HashNet加权平衡策略
```

## 使用方法

### 基本使用

**启用加权平衡策略训练**:
```bash
python demo.py --dataset mirflickr --hash_lens 16 --epoch 50 --task 1 --use_weighted_balance
```

**不使用加权平衡策略（原始方法）**:
```bash
python demo.py --dataset mirflickr --hash_lens 16 --epoch 50 --task 1
```

### 参数说明

- `--dataset`: 数据集名称 (mirflickr, mscoco, nus-wide)
- `--hash_lens`: 哈希码长度 (16, 32, 64等)
- `--epoch`: 训练轮数
- `--task`: 任务类型
  - 0: 训练实值特征
  - 1: 训练哈希码
  - 2: 测试实值特征
  - 3: 测试哈希码
- `--use_weighted_balance`: 启用HashNet加权平衡策略（flag参数）
- `--device`: CUDA设备 (默认: cuda:0)

### 批量测试脚本

提供了测试脚本 `test-weighted-balance.sh`:
```bash
chmod +x test-weighted-balance.sh
./test-weighted-balance.sh
```

该脚本会在不同数据集和配置下对比使用/不使用加权平衡策略的效果。

## 技术细节

### 加权平衡策略原理

HashNet的加权平衡策略通过以下方式处理类别不均衡：

1. **计算样本对相似度**:
   ```python
   similarity = (label1 @ label2.T > 0)
   ```

2. **计算损失**:
   ```python
   exp_loss = log(1 + exp(-|dot_product|)) + max(dot_product, 0) - similarity * dot_product
   ```

3. **应用加权**:
   ```python
   S1 = sum(positive_pairs)  # 正样本对数量
   S0 = sum(negative_pairs)  # 负样本对数量
   S = S0 + S1               # 总数量
   
   # 正样本权重
   exp_loss[positive] *= (S / S1)
   # 负样本权重
   exp_loss[negative] *= (S / S0)
   ```

### 优势

1. **自适应平衡**: 自动根据正负样本比例调整权重
2. **处理不均衡**: 在类别分布不均衡的数据集上表现更好
3. **保持兼容**: 可以通过参数控制，不影响原有功能

## 实验建议

1. **对比实验**: 建议在相同设置下对比使用/不使用加权平衡策略的效果
2. **不同数据集**: 在不同类别分布的数据集上测试（mirflickr, mscoco, nus-wide）
3. **不同哈希长度**: 测试不同哈希码长度（16, 32, 64 bits）

## 示例输出

训练时会输出：
```
=============== mirflickr--16 bits--Total epochs:50 ===============
...Training is beginning... 1
total_param: XXXXXX
epoch: 1
...
Testing...
I2T: 0.XXXX , T2I: 0.XXXX
```

## 注意事项

1. 确保数据集的label信息正确加载（已在代码中处理）
2. 加权平衡策略主要用于哈希码学习阶段（task=1）
3. 可以根据实验结果调整loss2的权重系数（当前为0.5）

## 参考

- HashNet论文: "HashNet: Deep Learning to Hash by Continuation"
- HashNet实现: `/HashNet/pytorch/src/loss.py` 中的 `pairwise_loss_updated`
