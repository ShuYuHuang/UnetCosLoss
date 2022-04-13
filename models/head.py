import torch
from torch import nn
import torch.nn.functional as F
def cosine(fts, prototypes, scaler=1):
    cos=torch.stack(
        [F.cosine_similarity(fts, p[None,..., None, None], dim=1) * scaler
            for p in prototypes]
    ,dim=1)
    return cos
class MetricLayer(nn.Module):
    def __init__(self, n_in_features,n_out_features=2,metric=cosine):
        super().__init__()
        self.weight = nn.Parameter(torch.Tensor(n_out_features, n_in_features))
        nn.init.xavier_uniform_(self.weight,gain=1.0)
        self.metric=metric
    def forward(self,x):
        return self.metric(x,self.weight)