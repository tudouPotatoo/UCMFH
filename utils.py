import torch
from os import path as osp

def save_checkpoints(self):
    """保存模型检查点"""
    if self.task == 0:
        file_name = self.dataset + '_fusion.pth'
        ckp_path = osp.join(self.model_dir, 'real', file_name)
        obj = {
            'ImageTransformer': self.ImageTransformer.state_dict(),
            'TextTransformer': self.TextTransformer.state_dict(),
            'CrossAttention': self.CrossAttention.state_dict()
        }
    elif self.task == 1:
        file_name = self.dataset + '_hash_' + str(self.nbits) + ".pth"
        ckp_path = osp.join(self.model_dir, 'hash', file_name)
        obj = {
            'ImageTransformer': self.ImageTransformer.state_dict(),
            'TextTransformer': self.TextTransformer.state_dict(),
            'CrossAttention': self.CrossAttention.state_dict(),
            'FusionMlp': self.FusionMlp.state_dict(),
            'ImageMlp': self.ImageMlp.state_dict(),
            'TextMlp': self.TextMlp.state_dict()
        }
    torch.save(obj, ckp_path)
    print(f'✅ Save the {"real" if self.task==0 else "hash"} model successfully to {ckp_path}')


def load_checkpoints(self, file_name):
    """加载模型检查点"""
    ckp_path = file_name
    try:
        obj = torch.load(ckp_path, map_location=self.device)
        print(f'✅ Load checkpoint from {ckp_path}')
    except IOError:
        print(f'❌ Fail to load checkpoint {ckp_path}!')
        raise IOError
    
    # 加载各个组件
    self.ImageTransformer.load_state_dict(obj['ImageTransformer'])
    self.TextTransformer.load_state_dict(obj['TextTransformer'])
    self.CrossAttention.load_state_dict(obj['CrossAttention'])
    
    if self.task == 3:  # test hash
        if 'FusionMlp' in obj:
            self.FusionMlp.load_state_dict(obj['FusionMlp'])
        if 'ImageMlp' in obj:
            self.ImageMlp.load_state_dict(obj['ImageMlp'])
        if 'TextMlp' in obj:
            self.TextMlp.load_state_dict(obj['TextMlp'])