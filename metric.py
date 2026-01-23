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


class WeightedBalanceLoss(nn.Module):
    """
    HashNet's weighted balance strategy for handling class imbalance.
    Based on pairwise_loss_updated from HashNet.
    """
    def __init__(self, device='cuda:0'):
        super(WeightedBalanceLoss, self).__init__()
        self.device = device
    
    def forward(self, outputs1, outputs2, label1, label2):
        """
        Args:
            outputs1: embeddings from first modality (image)
            outputs2: embeddings from second modality (text)  
            label1: labels for first modality
            label2: labels for second modality
        """
        # Calculate similarity matrix
        similarity = (torch.mm(label1.float(), label2.float().t()) > 0).float()
        
        # Calculate dot product
        dot_product = torch.mm(outputs1, outputs2.t())
        
        # Calculate loss with stable log-sum-exp trick
        exp_loss = torch.log(1 + torch.exp(-torch.abs(dot_product))) + \
                   torch.max(dot_product, torch.zeros_like(dot_product)) - \
                   similarity * dot_product
        
        # Apply weighted balance strategy
        mask_positive = similarity > 0
        mask_negative = similarity <= 0
        
        S1 = torch.sum(mask_positive.float())  # number of positive pairs
        S0 = torch.sum(mask_negative.float())  # number of negative pairs  
        S = S0 + S1  # total pairs
        
        # Weighted balance: give higher weight to minority class
        if S1 > 0:
            exp_loss[mask_positive] = exp_loss[mask_positive] * (S / S1)
        if S0 > 0:
            exp_loss[mask_negative] = exp_loss[mask_negative] * (S / S0)
        
        loss = torch.sum(exp_loss) / S
        
        return loss

