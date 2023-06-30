import torch
from torch import nn
from torch.nn import init
from config import CONFIG

class AggregationNet(nn.Module):

    def __init__(self) -> None:
        super(AggregationNet, self).__init__()
        self.theta = nn.parameter.Parameter(torch.randn(1, 2))
        nn.init.kaiming_uniform_(self.theta)

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        theta = torch.exp(self.theta) / torch.exp(self.theta).sum()
        out = theta[0][0] * x + theta[0][1] * y
        return out


class FusionNet(nn.Module):

    def __init__(self, out_features: int = 512) -> None:
        super(FusionNet, self).__init__()

        self.appearance_net = nn.Sequential(nn.Linear(in_features=CONFIG.MODEL.APP_FEATURE_DIM, out_features=out_features),
                                            nn.LeakyReLU())
        self.shape_net = nn.Sequential(nn.Linear(in_features=10, out_features=out_features),
                                       nn.LeakyReLU())

        self.theta = AggregationNet()
        self.bn = nn.BatchNorm1d(CONFIG.MODEL.AGG_FEATURE_DIM)
        init.normal_(self.bn.weight.data, 1.0, 0.02)
        init.constant_(self.bn.bias.data, 0.0),
        

    def forward(self, appearance_features: torch.Tensor,
                shape_features: torch.Tensor) -> torch.Tensor:
        appearance = self.appearance_net(appearance_features) # 512
        shape = self.shape_net(shape_features) # 512
        if CONFIG.MODEL.AGG == 'SUM':
            agg_features = self.theta(x=appearance, y=shape)
        else: 
            agg_features = torch.cat((appearance, shape),dim=1)
        agg_features = self.bn(agg_features)
        return agg_features
