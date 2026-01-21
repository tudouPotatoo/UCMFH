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
                print(f"\n{'='*60}")
                print(f"Epoch {epoch+1}/{self.total_epoch}")
                print(f"{'='*60}")
                train_loss = self.trainhash()
                if((epoch+1)%10==0):
                    print("→ 评估中...")
                    img2text, text2img = self.evaluate() 
                    I2T_MAP.append(img2text)
                    T2I_MAP.append(text2img)
                    print(f'✓ I2T: {img2text:.4f} | T2I: {text2img:.4f} | Avg: {(img2text+text2img)/2:.4f}')
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
            print(f"\n🔍 哈希码二值化统计:")
            
            # 二值化前的统计（紧凑格式）
            print(f"  Tanh输出: Query [{qu_BI.min():.3f}, {qu_BI.max():.3f}] | "
                  f"Retrieval [{re_BI.min():.3f}, {re_BI.max():.3f}]")
            
            qu_BI = torch.sign(torch.tensor(qu_BI)).cpu().numpy()
            qu_BT = torch.sign(torch.tensor(qu_BT)).cpu().numpy()
            re_BT = torch.sign(torch.tensor(re_BT)).cpu().numpy()
            re_BI = torch.sign(torch.tensor(re_BI)).cpu().numpy()
            
            # 二值化后的统计（紧凑格式）
            qu_ones = (qu_BI == 1).sum() / qu_BI.size * 100
            re_ones = (re_BI == 1).sum() / re_BI.size * 100
            print(f"  二值分布: Query +1:{qu_ones:.1f}% | Retrieval +1:{re_ones:.1f}%")
            
            # 仅展示1个样本示例（前16个bits）
            if len(qu_BI) > 0:
                print(f"  示例: {qu_BI[0][:16].tolist()}")
            print()
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
        """训练哈希函数（使用拼接特征 + 量化损失）"""
        self.ImageTransformer.train()
        self.TextTransformer.train()
        self.CrossAttention.train()
        self.FusionMlp.train()
        
        alpha = 0.1  # 量化损失权重
        running_loss = 0.0
        running_q_loss = 0.0
        running_contrast_loss = 0.0
        
        # 用于统计整个epoch的哈希码特性
        total_extreme_ratio = 0.0
        total_near_zero_ratio = 0.0
        num_batches = 0
        
        for idx, (img, txt, _,_) in enumerate(self.train_loader):
            img, txt = img.to(self.device), txt.to(self.device)
            
            # ✅ 第一阶段：单模态增强 + 跨模态融合
            img_enhanced = self.ImageTransformer(img)
            text_enhanced = self.TextTransformer(txt)
            img_embedding, text_embedding = self.CrossAttention(img_enhanced, text_enhanced)
            
            # 融合特征的对比损失
            loss1 = self.ContrastiveLoss(img_embedding, text_embedding)

            # ✅ 第二阶段：拼接特征并生成哈希码
            fused_feat = torch.cat([img_embedding, text_embedding], dim=1)
            fused_hash = self.FusionMlp(fused_feat)
            
            # ✅ 第三阶段：计算量化损失
            q_loss = torch.mean((fused_hash - torch.sign(fused_hash.detach()))**2)
            
            # 累计统计
            with torch.no_grad():
                extreme_ratio = ((fused_hash > 0.8) | (fused_hash < -0.8)).float().mean().item()
                near_zero_ratio = ((fused_hash > -0.1) & (fused_hash < 0.1)).float().mean().item()
                total_extreme_ratio += extreme_ratio
                total_near_zero_ratio += near_zero_ratio
                num_batches += 1
            
            # ✅ 总损失
            loss = loss1 + alpha * q_loss
            
            # 反向传播
            self.optimizer_Fusion.zero_grad()
            self.optimizer_Hash.zero_grad()
            loss.backward()
            self.optimizer_Fusion.step()
            self.optimizer_Hash.step()
            
            running_loss += loss.item()
            running_q_loss += q_loss.item()
            running_contrast_loss += loss1.item()
        
            self.Hash_scheduler.step()
        
        # ✅ Epoch总结（紧凑格式，单行显示）
        avg_loss = running_loss / len(self.train_loader)
        avg_q_loss = running_q_loss / len(self.train_loader)
        avg_contrast = running_contrast_loss / len(self.train_loader)
        avg_extreme = total_extreme_ratio / num_batches * 100
        avg_near_zero = total_near_zero_ratio / num_batches * 100
        
        print(f"📊 总损失:{avg_loss:.4f} | 对比:{avg_contrast:.4f} | 量化:{avg_q_loss:.4f} | "
              f"极端值:{avg_extreme:.1f}% | 近零值:{avg_near_zero:.1f}%")
        
        return avg_loss