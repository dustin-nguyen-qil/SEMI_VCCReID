import torch 
from torch import nn 

class DivRegLoss(nn.Module):
    def __init__(self, detach=True, sqrt=True):
        super(DivRegLoss, self).__init__()
        print('detach: {}'.format(detach))
        self.detach = detach
        self.sqrt = sqrt

    def forward_once(self, p1, p2):
        """p1: [bs, k], p2: [bs, k]
        """
        bs, k = p1.size()

        I = torch.eye(2, dtype=p1.dtype).cuda()
        x = torch.stack((p1, p2), 1) #[bs, 2, k]
        if self.sqrt:
            x = torch.sqrt(x)
        tmp = torch.bmm(x, x.transpose(1, 2)) #[bs, 2, 2]
        tmp = tmp - I.unsqueeze(0)
        tmp = tmp.view(bs, -1)
        tmp = torch.norm(tmp, dim=1) / tmp.size(1)
        loss = tmp.mean()
        return loss

    def forward(self, inputs):
        """inputs: [[bs, k], [bs, k], [bs, k]]
        """
        p1, p2, p3 = inputs
        if self.detach:
            p1 = p1.detach()
        loss1 = self.forward_once(p1, p2)
        loss2 = self.forward_once(p1, p3)
        loss3 = self.forward_once(p2, p3)
        return (loss1 + loss2 + loss3) / 3