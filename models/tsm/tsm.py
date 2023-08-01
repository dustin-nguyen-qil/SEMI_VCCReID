import os
import torch
import os.path as osp
import torch.nn as nn
from .regressor import Regressor
from .encoder import ft, TemporalEncoder

from config import CONFIG

TSM_DATA_PATH = 'work_space/tsm'

class TSM(nn.Module):
    def __init__(
            self,
            n_layers=1,
            hidden_size=2048,
            add_linear=False,
            bidirectional=False,
            use_residual=True,
            pretrained=osp.join(TSM_DATA_PATH, 'spin_model_checkpoint.pth.tar'),
    ):

        super(TSM, self).__init__()

        self.encoder = TemporalEncoder(
            n_layers=n_layers,
            hidden_size=hidden_size,
            bidirectional=bidirectional,
            add_linear=add_linear,
            use_residual=use_residual,
        )

        # self.ft = ft()
        # checkpoint = torch.load(pretrained, map_location='cpu')
        # self.ft.load_state_dict(checkpoint['model'], strict=False)

        # regressor can predict cam, pose and shape params in an iterative way
        self.regressor = Regressor()

        if pretrained and os.path.isfile(pretrained):
            pretrained_dict = torch.load(pretrained)['model']

            self.regressor.load_state_dict(pretrained_dict, strict=False)


    def forward(self, input, J_regressor=None):
        # input size NTF
        # batch_size, seqlen, nc, h, w = input.shape
        batch_size, seqlen, f = input.shape #f=2048

        # feature = self.ft(input.reshape(-1, nc, h, w))
        # feature = feature.reshape(batch_size, seqlen, -1)

        # feature = self.encoder(feature)
        feature = self.encoder(input)
        feature = feature.reshape(-1, feature.size(-1))
        smpl_output = self.regressor(feature, J_regressor=J_regressor)

        for s in smpl_output:
            s['theta'] = s['theta'].reshape(batch_size, seqlen, -1)
            s['shape_1024'] = s['shape_1024'].reshape(batch_size, seqlen, -1)
        
        output = smpl_output[-1]

        pred_shape_1024 = (output['shape_1024'].reshape(seqlen, -1)).detach()

        pred_beta = output['theta'][:, :, 75:].reshape(seqlen, -1).detach()

        return pred_beta, pred_shape_1024

