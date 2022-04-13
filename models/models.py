from torch import nn
import torch.nn.functional as F
class SegModel(nn.Module):
    def __init__(self,backbone,head,output_sizes=(128,128)):
        super().__init__()
        assert(backbone is not None)
        self.backbone=backbone
        self.head=head
        self.output_sizes=output_sizes
    def forward(self,data,label=None):
        # Transfer Learing: backbone+ output head
        hidden=self.backbone(data)
        features=self.head(hidden)
        if label:
            logits = F.interpolate(features, size=label.shape[-2:], mode='bilinear')
        else:
            logits = F.interpolate(features, size=self.output_sizes, mode='bilinear')
        return logits
