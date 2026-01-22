import torch
import torch.nn.functional as F
import torch.nn as nn 

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda:0', temperature=0.5):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.register_buffer("temperature", torch.tensor(temperature).to(device))			
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		
    def forward(self, emb_i, emb_j):		
        z_i = F.normalize(emb_i, dim=1)     # (bs, dim)  --->  (bs, dim)
        z_j = F.normalize(emb_j, dim=1)     # (bs, dim)  --->  (bs, dim)
        representations = torch.cat([z_i, z_j], dim=0)          # repre: (2*bs, dim)
        similarity_matrix = F.cosine_similarity(representations.unsqueeze(1), representations.unsqueeze(0), dim=2)      # simi_mat: (2*bs, 2*bs)
        sim_ij = torch.diag(similarity_matrix, self.batch_size)         # bs
        sim_ji = torch.diag(similarity_matrix, -self.batch_size)        # bs
        positives = torch.cat([sim_ij, sim_ji], dim=0)                  # 2*bs
        nominator = torch.exp(positives / self.temperature)             # 2*bs
        denominator = self.negatives_mask * torch.exp(similarity_matrix / self.temperature)             # 2*bs, 2*bs
        loss_partial = -torch.log(nominator / torch.sum(denominator, dim=1))        # 2*bs
        loss = torch.sum(loss_partial) / (2 * self.batch_size)
        return loss


class ContrastiveLossBalanced(nn.Module):
    """
    带加权平衡的对比学习损失函数
    
    借鉴HashNet的pairwise_loss_updated思想，对正负样本对进行加权平衡，
    避免模型因负样本过多而只学会"都不相似"的策略。
    
    核心改进：
    1. 自动计算正负样本对的数量
    2. 给稀有的正样本对更大的权重
    3. 给常见的负样本对较小的权重
    
    这样可以强制模型更加关注"什么是相似的"，而不是简单记住"大部分都不相似"。
    """
    def __init__(self, batch_size, device='cuda:0', temperature=0.5):
        super(ContrastiveLossBalanced, self).__init__()
        self.batch_size = batch_size
        self.device = device
        self.register_buffer("temperature", torch.tensor(temperature).to(device))
        
        # 创建正负样本的mask
        # 对于跨模态检索：正样本是对角线（配对的图文）
        # 负样本是非对角线（不配对的图文）
        identity = torch.eye(batch_size, dtype=bool).to(device)
        self.register_buffer("positive_mask", identity)  # [bs, bs]
        self.register_buffer("negative_mask", ~identity)  # [bs, bs]
        
    def forward(self, emb_i, emb_j):
        """
        Args:
            emb_i: 图像嵌入 [batch_size, dim]
            emb_j: 文本嵌入 [batch_size, dim]
        Returns:
            loss: 加权平衡后的对比学习损失
        """
        # L2归一化
        z_i = F.normalize(emb_i, dim=1)  # [bs, dim]
        z_j = F.normalize(emb_j, dim=1)  # [bs, dim]
        
        # 计算图像-文本的相似度矩阵
        # similarity_matrix[i,j] = 第i个图像与第j个文本的相似度
        similarity_matrix = torch.mm(z_i, z_j.t())  # [bs, bs]
        
        # 🆕 计算正负样本对的数量
        S1 = self.positive_mask.sum().float()  # 正样本对数量 = batch_size
        S0 = self.negative_mask.sum().float()  # 负样本对数量 = batch_size * (batch_size - 1)
        S = S1 + S0  # 总样本对数量 = batch_size^2
        
        # 计算缩放后的相似度
        scaled_sim = similarity_matrix / self.temperature  # [bs, bs]
        
        # 对于每个图像，计算与所有文本的InfoNCE损失
        # exp_sim[i,j] = exp(sim[i,j] / temp)
        exp_sim = torch.exp(scaled_sim)  # [bs, bs]
        
        # 对于每个图像i：
        # - 正样本：与它配对的文本j=i
        # - 负样本：其他所有文本j≠i
        # InfoNCE loss = -log(exp(sim[i,i]) / sum_j(exp(sim[i,j])))
        
        # 提取正样本的相似度（对角线元素）
        positive_sim = torch.diag(scaled_sim)  # [bs]
        
        # 分母：所有样本对的exp(similarity)之和
        # denominator[i] = sum_j exp(sim[i,j])
        denominator = exp_sim.sum(dim=1)  # [bs]
        
        # 原始的InfoNCE损失（未加权）
        # loss[i] = -log(exp(sim[i,i]) / denominator[i])
        #         = -sim[i,i] + log(denominator[i])
        raw_loss = -positive_sim + torch.log(denominator)  # [bs]
        
        # 🆕 关键改进：分别计算正负样本的贡献，并加权
        # 对于每个图像i，分解损失：
        # loss[i] = -log(exp(positive) / (exp(positive) + sum_negative exp))
        #         = -log(1 / (1 + sum_negative exp / exp(positive)))
        #         = log(1 + sum_negative exp / exp(positive))
        
        # 但为了简化，我们采用HashNet的加权策略：
        # 直接对最终的loss加权
        
        # 方法：计算每个样本对的损失贡献，然后分别对正负样本加权
        # 但在标准InfoNCE中，这不太直观
        # 所以我们采用另一种等价方式：
        
        # 计算正样本的损失贡献（使用正样本权重）
        positive_weight = S / S1  # 权重 = 总数 / 正样本数
        
        # 为了加权负样本，我们需要重新计算损失
        # 新的损失 = -log(exp(pos) / (exp(pos) + weighted_sum(exp(neg))))
        
        # 计算加权后的负样本和
        exp_positive = torch.exp(positive_sim)  # [bs]
        exp_negative_sum = denominator - exp_positive  # [bs] 负样本的exp和
        
        # 应用负样本权重
        negative_weight = S / S0  # 权重 = 总数 / 负样本数
        weighted_exp_negative = exp_negative_sum * negative_weight
        
        # 加权后的损失
        # loss = -log(exp(pos) / (exp(pos) + weighted_exp(neg)))
        weighted_loss = -torch.log(exp_positive / (exp_positive + weighted_exp_negative))
        
        # 再对正样本应用额外的权重（因为我们希望更关注正样本）
        final_loss = weighted_loss * positive_weight
        
        # 返回平均损失
        return final_loss.mean()


class PairwiseLoss(nn.Module):
    """
    直接借鉴HashNet的pairwise_loss_updated
    
    这个损失函数需要标签信息来判断样本对是否相似。
    适用于有监督的哈希学习场景。
    """
    def __init__(self, device='cuda:0'):
        super(PairwiseLoss, self).__init__()
        self.device = device
    
    def forward(self, outputs1, outputs2, label1, label2):
        """
        Args:
            outputs1: 第一批样本的哈希码/特征 [batch_size, hash_dim]
            outputs2: 第二批样本的哈希码/特征 [batch_size, hash_dim]
            label1: 第一批样本的标签 [batch_size, num_classes]
            label2: 第二批样本的标签 [batch_size, num_classes]
        
        Returns:
            loss: 加权平衡后的成对损失
        """
        # 计算标签相似度：如果两个样本有共同的标签，则相似
        # similarity[i,j] = 1 if label1[i] 和 label2[j] 有重叠，否则为0
        similarity = (torch.mm(label1.float(), label2.float().t()) > 0).float()
        
        # 计算哈希码的内积
        dot_product = torch.mm(outputs1, outputs2.t())  # [bs, bs]
        
        # 创建正负样本mask
        mask_positive = similarity > 0
        mask_negative = similarity <= 0
        
        # 计算损失（使用log-exp技巧保证数值稳定）
        # loss = log(1 + exp(-|dot_product|)) + max(dot_product, 0) - similarity * dot_product
        exp_loss = (torch.log(1 + torch.exp(-torch.abs(dot_product))) + 
                   torch.max(dot_product, torch.zeros_like(dot_product)) - 
                   similarity * dot_product)
        
        # 🆕 加权平衡（HashNet的核心思想）
        S1 = mask_positive.sum().float()  # 正样本对数量
        S0 = mask_negative.sum().float()  # 负样本对数量
        S = S1 + S0  # 总样本对数量
        
        # 对正负样本分别加权
        weighted_loss = torch.zeros_like(exp_loss)
        weighted_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        weighted_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)
        
        # 返回平均损失
        return weighted_loss.sum() / S

