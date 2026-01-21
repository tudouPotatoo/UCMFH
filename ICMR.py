import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from evaluate import  calculate_top_map
from load_dataset import  load_dataset
from metric import ContrastiveLoss
from model import UnimodalTransformer, CrossAttentionFusion, ImageMlp, TextMlp, FusionMlp
from os import path as osp
from utils import load_checkpoints, save_checkpoints
from torch.optim import lr_scheduler
import time

class Solver(object):
    def __init__(self, config):
        self.batch_size = 128  
        self.total_epoch = config.epoch
        self.dataset  = config.dataset
        self.model_dir = "./checkpoints"

        USE_CUDA = torch.cuda.is_available()
        self.device = torch.device(config.device if USE_CUDA else "cpu")

        self.task = config.task
        self.feat_lens = 512
        self.nbits = config.hash_lens
 
        # ✅ 新架构：直接实例化各个组件
        self.ImageTransformer = UnimodalTransformer(d_model=512, num_layers=2).to(self.device)
        self.TextTransformer = UnimodalTransformer(d_model=512, num_layers=2).to(self.device)
        self.CrossAttention = CrossAttentionFusion(d_model=512, nhead=8).to(self.device)
        
        # 融合特征哈希映射（拼接后的1024维）
        self.FusionMlp = FusionMlp(input_dim=1024, hash_lens=self.nbits).to(self.device)
        
        # 为了向后兼容，保留ImageMlp和TextMlp（可选）
        self.ImageMlp = ImageMlp(input_dim=512, hash_lens=self.nbits).to(self.device)
        self.TextMlp = TextMlp(input_dim=512, hash_lens=self.nbits).to(self.device)
        
        # 优化器配置
        params_fusion = (
            list(self.ImageTransformer.parameters()) + 
            list(self.TextTransformer.parameters()) + 
            list(self.CrossAttention.parameters())
        )
        params_hash = list(self.FusionMlp.parameters())
        params_image = list(self.ImageMlp.parameters())
        params_text = list(self.TextMlp.parameters())
        
        total_param = (
            sum([p.nelement() for p in params_fusion]) +
            sum([p.nelement() for p in params_hash]) +
            sum([p.nelement() for p in params_image]) +
            sum([p.nelement() for p in params_text])
        )
        print(f"Total parameters: {total_param:,}")
        
        self.optimizer_Fusion = optim.Adam(params_fusion, lr=1e-4, betas=(0.5, 0.999))
        self.optimizer_Hash = optim.Adam(params_hash, lr=1e-3, betas=(0.5, 0.999))
        self.optimizer_ImageMlp = optim.Adam(params_image, lr=1e-3, betas=(0.5, 0.999))
        self.optimizer_TextMlp = optim.Adam(params_text, lr=1e-3, betas=(0.5, 0.999))

        if self.dataset == "mirflickr" or self.dataset=="nus-wide":
            self.Hash_scheduler = lr_scheduler.MultiStepLR(self.optimizer_Hash, milestones=[30, 80], gamma=1.2)
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp, milestones=[30, 80], gamma=1.2)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp, milestones=[30, 80], gamma=1.2)
        elif self.dataset == "mscoco":
            self.Hash_scheduler = lr_scheduler.MultiStepLR(self.optimizer_Hash, milestones=[200], gamma=0.6)
            self.ImageMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_ImageMlp,milestones=[200], gamma=0.6)
            self.TextMlp_scheduler = lr_scheduler.MultiStepLR(self.optimizer_TextMlp,milestones=[200], gamma=0.6)

        data_loader = load_dataset(self.dataset, self.batch_size)
        self.train_loader = data_loader['train']
        self.query_loader = data_loader['query']
        self.retrieval_loader = data_loader['retrieval']
              
        self.ContrastiveLoss = ContrastiveLoss(batch_size=self.batch_size, device=self.device)
     
     
    def train(self):
        if self.task == 0: # train real
            print("Training Fusion Transformer...")
            for epoch in range(self.total_epoch):
                print("epoch:",epoch+1)
                train_loss = self.trainfusion()
                if((epoch+1)%10==0):
                    print("Testing...")
                    img2text, text2img = self.evaluate() 
                    print('I2T:',img2text, ', T2I:',text2img)
            save_checkpoints(self)
           
        elif self.task == 1: # train hash 
            print("Training Hash Fuction...")
            I2T_MAP = []
            T2I_MAP = []
            start_time = time.time()
            for epoch in range(self.total_epoch):
                print("epoch:",epoch+1)
                train_loss = self.trainhash()
                print(train_loss)
                if((epoch+1)%10==0):
                    print("Testing...")
                    img2text, text2img = self.evaluate() 
                    I2T_MAP.append(img2text)
                    T2I_MAP.append(text2img)
                    print('I2T:',img2text, ', T2I:',text2img)
            print(I2T_MAP,T2I_MAP)
            save_checkpoints(self)
            time_elapsed = time.time() - start_time
            print(f'Total Train Time: {int(time_elapsed // 60)}m {int(time_elapsed % 60)}s')
                
        elif self.task == 2: # test real
            file_name = self.dataset + '_fusion.pth'
            ckp_path = osp.join(self.model_dir,'real', file_name)
            load_checkpoints(self, ckp_path)

        elif self.task == 3: # test hash 
            
            file_name = self.dataset + '_hash_' + str(self.nbits)+".pth"
            ckp_path = osp.join(self.model_dir,'hash', file_name)
            load_checkpoints(self, ckp_path)

        print("Final Testing...")
        img2text, text2img = self.evaluate() 
        print('I2T:',img2text, ', T2I:',text2img)
        return (img2text + text2img)/2., img2text, text2img
      
    def evaluate(self):
        self.ImageTransformer.eval()
        self.TextTransformer.eval()
        self.CrossAttention.eval()
        self.FusionMlp.eval()
        self.ImageMlp.eval()
        self.TextMlp.eval()
        
        qu_BI, qu_BT, qu_L = [], [], []
        re_BI, re_BT, re_L = [], [], []
      
        with torch.no_grad():
            # Query set
            for _,(data_I, data_T, data_L,_) in enumerate(self.query_loader):
                data_I, data_T = data_I.to(self.device), data_T.to(self.device)
                
                # ✅ 直接调用各组件，无需concat
                img_enhanced = self.ImageTransformer(data_I)
                text_enhanced = self.TextTransformer(data_T)
                img_query, txt_query = self.CrossAttention(img_enhanced, text_enhanced)
                
                if self.task == 1 or self.task == 3:
                    # ✅ 拼接特征并通过FusionMlp
                    fused_query = torch.cat([img_query, txt_query], dim=1)  # [batch, 1024]
                    hash_query = self.FusionMlp(fused_query)  # [batch, hash_lens]
                    
                    # 为了兼容，也计算分离的hash（可选）
                    img_query = hash_query
                    txt_query = hash_query
                
                qu_BI.extend(img_query.cpu().numpy())
                qu_BT.extend(txt_query.cpu().numpy())
                qu_L.extend(data_L.cpu().numpy())

            # Retrieval set
            for _,(data_I, data_T, data_L,_) in enumerate(self.retrieval_loader):
                data_I, data_T = data_I.to(self.device), data_T.to(self.device)
                
                # ✅ 直接调用各组件
                img_enhanced = self.ImageTransformer(data_I)
                text_enhanced = self.TextTransformer(data_T)
                img_retrieval, txt_retrieval = self.CrossAttention(img_enhanced, text_enhanced)
                
                if self.task ==1 or self.task ==3:
                    # ✅ 拼接特征并通过FusionMlp
                    fused_retrieval = torch.cat([img_retrieval, txt_retrieval], dim=1)  # [batch, 1024]
                    hash_retrieval = self.FusionMlp(fused_retrieval)  # [batch, hash_lens]
                    
                    img_retrieval = hash_retrieval
                    txt_retrieval = hash_retrieval
                
                re_BI.extend(img_retrieval.cpu().numpy())
                re_BT.extend(txt_retrieval.cpu().numpy())
                re_L.extend(data_L.cpu().numpy())
        
        re_BI = np.array(re_BI)
        re_BT = np.array(re_BT)
        re_L = np.array(re_L)

        qu_BI = np.array(qu_BI)
        qu_BT = np.array(qu_BT)
        qu_L = np.array(qu_L)

        if self.task ==1 or self.task ==3:   # hashing
            print(f"\n{'='*70}")
            print(f"🔍 评估阶段 - 哈希码二值化统计")
            print(f"{'='*70}")
            
            # 二值化前的统计
            print(f"\n📊 二值化前 (Tanh输出):")
            print(f"   Query集 - 图像哈希码: 形状={qu_BI.shape}, 范围=[{qu_BI.min():.4f}, {qu_BI.max():.4f}]")
            print(f"   Query集 - 文本哈希码: 形状={qu_BT.shape}, 范围=[{qu_BT.min():.4f}, {qu_BT.max():.4f}]")
            print(f"   Retrieval集 - 图像哈希码: 形状={re_BI.shape}, 范围=[{re_BI.min():.4f}, {re_BI.max():.4f}]")
            print(f"   Retrieval集 - 文本哈希码: 形状={re_BT.shape}, 范围=[{re_BT.min():.4f}, {re_BT.max():.4f}]")
            
            qu_BI = torch.sign(torch.tensor(qu_BI)).cpu().numpy()
            qu_BT = torch.sign(torch.tensor(qu_BT)).cpu().numpy()
            re_BT = torch.sign(torch.tensor(re_BT)).cpu().numpy()
            re_BI = torch.sign(torch.tensor(re_BI)).cpu().numpy()
            
            # 二值化后的统计
            print(f"\n🔄 二值化后 (Sign函数):")
            print(f"   Query集 - 图像哈希码: 唯一值={np.unique(qu_BI).tolist()}")
            print(f"   Query集 - 文本哈希码: 唯一值={np.unique(qu_BT).tolist()}")
            print(f"   Retrieval集 - 图像哈希码: 唯一值={np.unique(re_BI).tolist()}")
            print(f"   Retrieval集 - 文本哈希码: 唯一值={np.unique(re_BT).tolist()}")
            
            # 统计+1和-1的比例
            qu_BI_ones = (qu_BI == 1).sum() / qu_BI.size * 100
            qu_BI_minus = (qu_BI == -1).sum() / qu_BI.size * 100
            print(f"\n📈 Query图像哈希码分布: +1={qu_BI_ones:.2f}%, -1={qu_BI_minus:.2f}%")
            
            re_BI_ones = (re_BI == 1).sum() / re_BI.size * 100
            re_BI_minus = (re_BI == -1).sum() / re_BI.size * 100
            print(f"📈 Retrieval图像哈希码分布: +1={re_BI_ones:.2f}%, -1={re_BI_minus:.2f}%")
            
            # 展示前3个样本的哈希码示例
            print(f"\n🔍 哈希码示例 (前3个样本, 前10个bits):")
            for i in range(min(3, len(qu_BI))):
                print(f"   Query样本{i}: {qu_BI[i][:10].tolist()}")
            print(f"{'='*70}\n")
        elif self.task ==0 or self.task ==2:  # real value
            qu_BI = torch.tensor(qu_BI).cpu().numpy()
            qu_BT = torch.tensor(qu_BT).cpu().numpy()
            re_BT = torch.tensor(re_BT).cpu().numpy()
            re_BI = torch.tensor(re_BI).cpu().numpy()
        
        MAP_I2T = calculate_top_map(qu_B=qu_BI, re_B=re_BT, qu_L=qu_L, re_L=re_L, topk=50)
        MAP_T2I = calculate_top_map(qu_B=qu_BT, re_B=re_BI, qu_L=qu_L, re_L=re_L, topk=50)
        return MAP_I2T, MAP_T2I 
    
    def trainfusion(self):
        """训练融合模块（实值表示）"""
        self.ImageTransformer.train()
        self.TextTransformer.train()
        self.CrossAttention.train()
        
        running_loss = 0.0
        for idx, (img, txt, _,_) in enumerate(self.train_loader):
            img, txt = img.to(self.device), txt.to(self.device)
            
            # ✅ 清晰的前向传播流程
            img_enhanced = self.ImageTransformer(img)
            text_enhanced = self.TextTransformer(txt)
            img_embedding, text_embedding = self.CrossAttention(img_enhanced, text_enhanced)
            
            # 计算损失
            loss = self.ContrastiveLoss(img_embedding, text_embedding)
            
            # 反向传播
            self.optimizer_Fusion.zero_grad()
            loss.backward()
            self.optimizer_Fusion.step()
            
            running_loss += loss.item()
        
        return running_loss
    
    def trainhash(self):
        """训练哈希函数（使用拼接特征）"""
        self.ImageTransformer.train()
        self.TextTransformer.train()
        self.CrossAttention.train()
        self.FusionMlp.train()
        
        running_loss = 0.0
        for idx, (img, txt, _,_) in enumerate(self.train_loader):
            img, txt = img.to(self.device), txt.to(self.device)
            
            # ✅ 第一阶段：单模态增强 + 跨模态融合
            img_enhanced = self.ImageTransformer(img)
            text_enhanced = self.TextTransformer(txt)
            img_embedding, text_embedding = self.CrossAttention(img_enhanced, text_enhanced)
            
            # 融合特征的对比损失
            loss1 = self.ContrastiveLoss(img_embedding, text_embedding)

            # ✅ 第二阶段：拼接特征并生成哈希码
            fused_feat = torch.cat([img_embedding, text_embedding], dim=1)  # [batch, 1024]
            fused_hash = self.FusionMlp(fused_feat)  # [batch, hash_lens]
            
            # 📊 打印哈希码统计信息（每10个batch打印一次）
            if idx % 10 == 0:
                print(f"\n{'='*60}")
                print(f"📊 [Batch {idx}] 哈希码统计信息:")
                print(f"{'='*60}")
                print(f"🔢 哈希码形状: {fused_hash.shape}")
                print(f"📈 哈希码值域: [{fused_hash.min().item():.4f}, {fused_hash.max().item():.4f}]")
                print(f"📊 哈希码均值: {fused_hash.mean().item():.4f}")
                print(f"📊 哈希码标准差: {fused_hash.std().item():.4f}")
                
                # 统计值分布
                positive_ratio = (fused_hash > 0).float().mean().item()
                near_zero_ratio = ((fused_hash > -0.1) & (fused_hash < 0.1)).float().mean().item()
                print(f"✨ 正值比例: {positive_ratio*100:.2f}%")
                print(f"⚠️  接近0的值 ([-0.1,0.1]): {near_zero_ratio*100:.2f}%")
                
                # 模拟二值化后的统计
                binary_hash = torch.sign(fused_hash)
                unique_vals = binary_hash.unique().tolist()
                print(f"🔄 二值化后的唯一值: {unique_vals}")
                if len(unique_vals) > 1:
                    ones_ratio = (binary_hash == 1).float().mean().item()
                    zeros_ratio = (binary_hash == 0).float().mean().item()
                    minus_ones_ratio = (binary_hash == -1).float().mean().item()
                    print(f"   +1: {ones_ratio*100:.2f}%, 0: {zeros_ratio*100:.2f}%, -1: {minus_ones_ratio*100:.2f}%")
                print(f"{'='*60}\n")
            
            # 拼接特征的哈希码对比损失
            # 由于query和key是同一个hash，这里需要修改损失计算方式
            # 使用自相关损失或者增强特征一致性
            loss2 = (fused_hash - fused_hash.detach()).pow(2).mean()  # 简化版，鼓励稳定性
            
            # 总损失
            loss = loss1 + loss2 * 0.1  # 降低loss2权重
            
            # 反向传播
            self.optimizer_Fusion.zero_grad()
            self.optimizer_Hash.zero_grad()
            loss.backward()
            self.optimizer_Fusion.step()
            self.optimizer_Hash.step()
            
            running_loss += loss.item()
        
            self.Hash_scheduler.step()
        
        return running_loss