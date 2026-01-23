import torch
import torch.nn.functional as F
import torch.nn as nn 

class ContrastiveLoss(nn.Module):
    def __init__(self, batch_size, device='cuda:0', temperature=0.5, use_weighted_balance=False):
        super(ContrastiveLoss, self).__init__()
        self.batch_size = batch_size
        self.use_weighted_balance = use_weighted_balance
        self.register_buffer("temperature", torch.tensor(temperature).to(device))			
        self.register_buffer("negatives_mask", (~torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device)).float())		
        self.register_buffer("positives_mask", torch.eye(batch_size * 2, batch_size * 2, dtype=bool).to(device))
        # 创建对角线偏移的mask用于标识正样本对
        positive_pairs_mask = torch.zeros(batch_size * 2, batch_size * 2, dtype=bool).to(device)
        for i in range(batch_size):
            positive_pairs_mask[i, i + batch_size] = True
            positive_pairs_mask[i + batch_size, i] = True
        self.register_buffer("positive_pairs_mask", positive_pairs_mask)
        
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
        
        if self.use_weighted_balance:
            # HashNet加权平衡策略：根据正负样本数量比例进行加权
            num_positives = 2 * self.batch_size  # 正样本对数量
            num_negatives = 2 * self.batch_size * (2 * self.batch_size - 1) - num_positives  # 负样本对数量
            total_samples = num_positives + num_negatives
            
            # 计算加权系数：样本少的类别获得更大的权重
            positive_weight = total_samples / num_positives
            # negative_weight = total_samples / num_negatives
            
            # 对正样本的损失进行加权（增强正样本的重要性）
            loss = torch.sum(loss_partial * positive_weight) / (2 * self.batch_size)
        else:
            loss = torch.sum(loss_partial) / (2 * self.batch_size)
        
        return loss

