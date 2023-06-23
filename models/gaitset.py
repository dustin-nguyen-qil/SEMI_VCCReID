import copy

import torch
import torch.nn as nn

from models.utils.layers import (BasicConv2d, HorizontalPoolingPyramid,
                     PackSequenceWrapper, SeparateFCs, SetBlockWrapper)


class GaitSet(nn.Module):
    """
        GaitSet: Regarding Gait as a Set for Cross-View Gait Recognition
        Arxiv:  https://arxiv.org/abs/1811.06186
        Github: https://github.com/AbnerHqC/GaitSet
    """

    def __init__(self, model_cfg) -> None:
        super(GaitSet, self).__init__()
        self.model_cfg = model_cfg
        sequence_length = torch.IntTensor(model_cfg.SEQUENCE_LENGTH)
        self.register_buffer('sequence_length', sequence_length)
        print(self.sequence_length.data.numpy().tolist())
        self.build_network(model_cfg=model_cfg)

    def build_network(self, model_cfg):
        in_c = model_cfg.IN_CHANNELS
        self.set_block1 = nn.Sequential(BasicConv2d(in_c[0], in_c[1], 5, 1, 2),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[1], in_c[1], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block2 = nn.Sequential(BasicConv2d(in_c[1], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[2], in_c[2], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        nn.MaxPool2d(kernel_size=2, stride=2))

        self.set_block3 = nn.Sequential(BasicConv2d(in_c[2], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True),
                                        BasicConv2d(in_c[3], in_c[3], 3, 1, 1),
                                        nn.LeakyReLU(inplace=True))

        self.gl_block2 = copy.deepcopy(self.set_block2)
        self.gl_block3 = copy.deepcopy(self.set_block3)

        self.set_block1 = SetBlockWrapper(self.set_block1)
        self.set_block2 = SetBlockWrapper(self.set_block2)
        self.set_block3 = SetBlockWrapper(self.set_block3)

        self.set_pooling = PackSequenceWrapper(pooling_func=torch.max)

        self.head = SeparateFCs(
            parts_num=model_cfg.SEPARATE_FC.PART_NUM,
            in_channels=model_cfg.SEPARATE_FC.IN_CHANNELS,
            out_channels=model_cfg.SEPARATE_FC.OUT_CHANNELS)

        self.hpp = HorizontalPoolingPyramid(bin_num=model_cfg.BIN_NUM)

    def forward(self, silhouettes: torch.Tensor) -> torch.Tensor:
        if len(silhouettes.size()) == 4:
            silhouettes = silhouettes.unsqueeze(2)

        outs = self.set_block1(silhouettes)
        gl = self.set_pooling(outs, self.sequence_length, dim=1)[0]
        gl = self.gl_block2(gl)

        outs = self.set_block2(outs)
        gl = gl + self.set_pooling(outs, self.sequence_length, dim=1)[0]
        gl = self.gl_block3(gl)

        outs = self.set_block3(outs)
        outs = self.set_pooling(outs, self.sequence_length, dim=1)[0]
        gl = gl + outs

        # Horizontal Pooling Matching, HPM
        feature1 = self.hpp(outs)  # [n, c, p]
        feature2 = self.hpp(gl)  # [n, c, p]
        feature = torch.cat([feature1, feature2], -1)  # [n, c, p]
        feature = feature.permute(2, 0, 1).contiguous()  # [p, n, c]
        embs = self.head(feature)
        embs = embs.permute(1, 0, 2).contiguous()  # [n, p, c]

        return embs
