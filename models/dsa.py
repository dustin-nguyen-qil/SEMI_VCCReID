import torch
from torch import nn
from torch.nn import functional as F

"""
    Difference-aware Shape Aggregation
    from a sequence of xc -> feed to DSA to get 10-D sequence-wise 3D shape
"""
class ShapeEncoder(nn.Module):
    def __init__(self, hidden):
        super(ShapeEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(in_features=1024, out_features=hidden),
            nn.Linear(in_features=hidden, out_features=128),
            nn.Linear(in_features=128, out_features=10)
        )
    def forward(self, x):
        x = self.encoder(x)
        return x 

class IntraFrameDiff(nn.Module):

    def __init__(self, num_frames: int) -> None:
        super(IntraFrameDiff, self).__init__()
        self.num_frames = num_frames
        self.conv = nn.Conv2d(in_channels=1,
                              out_channels=10,
                              kernel_size=(1, 10))

    def forward(self, beta_diff: torch.Tensor) -> torch.Tensor:
        """
        forward

        Args:
            beta_diff (torch.Tensor): Shape differences between frame
            (B, C x T x 10), with B is batch_size, T is the number of frames

        Returns:
            torch.Tensor: Intra-frame difference
        """
        B = beta_diff.size(0)
        x = self.conv(beta_diff)
        x = torch.sigmoid(x)
        return x.view(B, 1, self.num_frames, 10)


class InterFrameDiff(nn.Module):

    def __init__(self,
                 num_frames: int,
                 num_shape_parameters: int = 10) -> None:
        super(InterFrameDiff, self).__init__()
        self.num_frames = num_frames
        self.num_shape_params = num_shape_parameters
        for k in range(self.num_shape_params):
            __cnn = nn.Conv2d(in_channels=num_frames,
                              out_channels=1,
                              kernel_size=(1, 1),
                              stride=1)
            setattr(self, f"conv_{k+1}", __cnn)

    def forward(self, beta_diff: torch.Tensor):
        """
        forward

        Args:
            beta_diff (torch.Tensor): Shape differences between frame
            (B x T x 10), with B is batch_size, T is the number of frames

        Returns:
            torch.Tensor: Intra-frame difference
        """
        inter_beta_diff = beta_diff.transpose(dim0=2, dim1=1) - beta_diff
        B = inter_beta_diff.size(0)
        x = inter_beta_diff.reshape(10, B, self.num_frames, self.num_frames, 1)
        out = []
        for k in range(self.num_shape_params):
            module = getattr(self, f"conv_{k+1}")
            __x = module(x[k])
            __x = torch.sigmoid(__x)
            out.append(__x)
            # print(x.shape)
        out = torch.stack(tensors=out, dim=1)
        out = out.view(B, 1, self.num_frames, 10)
        return out


class DSA(nn.Module):

    def __init__(self,
                 num_frames: int = 8,
                 num_shape_parameters: int = 10) -> None:
        super(DSA, self).__init__()
        self.num_frames = num_frames
        self.num_shape_params = num_shape_parameters
        self.encoder = ShapeEncoder(512)
        self.intra_net = IntraFrameDiff(num_frames=self.num_frames)
        self.inter_net = InterFrameDiff(
            num_frames=self.num_frames,
            num_shape_parameters=num_shape_parameters)

    def forward(self, xc: torch.Tensor) -> torch.Tensor:
        """
        forward

        Args:
            xc (torch.Tensor): (B x T x 1024)

        Returns:
            torch.Tensor
        """
        beta = self.encoder(xc)
        beta_mean = torch.mean(input=beta, dim=0)
        beta_diff = beta - beta_mean
        wd = self.intra_net.forward(beta_diff=beta_diff.unsqueeze(1))
        wt = self.inter_net.forward(beta_diff=beta_diff.unsqueeze(1))
        ws = wd * wt
        ws = torch.softmax(ws, dim=2)
        beta_s = beta * ws
        beta_s = torch.sum(beta_s, dim=1)
        return beta, beta_s
