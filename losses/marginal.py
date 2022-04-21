import torch
from torch import nn
import torch.nn.functional as F
from .focal import FocalLoss

class AddMarginLoss(nn.Module):
    def __init__(self,ways, s=15.0, m=0.40,loss_fn=FocalLoss,*args,**kwargs):
        super().__init__()
        self.s = s
        self.m = m
        self.loss_fn=loss_fn(*args,**kwargs)
        self.ways=ways
    def forward(self, cosine, label=None):
        # 扣掉對cosine的margin
        cos_phi = cosine - self.m
        # 將onehot沒選中的類別不套用margin，onehot選中的套用margin     
        one_hot=F.one_hot(label,self.ways).transpose(-1,-2).transpose(-2,1).to(torch.float32)
        metric = (one_hot * cos_phi) + ((1.0 - one_hot) * cosine)
        # 將輸出對比放大
        metric *= self.s
        return self.loss_fn(metric,label)