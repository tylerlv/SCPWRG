from .unet_parts import *
import torch.nn as nn
from .GCN import *
import torch
import os

class PEncoder(nn.Module):
    def __init__(self, n_channel, basic_channel=16):
        super(PEncoder, self).__init__()
        self.n_channel = n_channel

        self.basic_channel = basic_channel

        self.db_0 = block(self.n_channel, basic_channel)
        self.db_1 = down_block(basic_channel, basic_channel*2)
        self.db_2 = down_block(basic_channel*2, basic_channel*4)
        self.db_3 = down_block(basic_channel*4, basic_channel*8)
        self.db_4 = down_block(basic_channel*8, basic_channel*16)

        self.gc = GraphConvolution(basic_channel*16*2*2, basic_channel*16*2*2)

    def forward(self, x, adj):
        d0 = self.db_0(x)
        d1 = self.db_1(d0)
        d2 = self.db_2(d1)
        d3 = self.db_3(d2)
        d4 = self.db_4(d3)

        d4_flatten = d4.view([d4.shape[0], -1])
        d4_gcn = self.gc(d4_flatten, adj)
        d4_refined = d4_gcn.view([-1, self.basic_channel*16, int(32/16), int(32/16)])
        whole = self.combine_patch(d4_refined)
        whole = whole.view([-1,whole.shape[0], whole.shape[1], whole.shape[2]])

        return d0, d1, d2, d3, d4_refined, whole

    def combine_patch(self, patch):
        whole = torch.cat([patch[0], patch[1], patch[2], patch[3], patch[4], patch[5], patch[6], patch[7]],dim=2)
        for i in range(1,8):
            whole_part = torch.cat([patch[i*8+0], patch[i*8+1], patch[i*8+2], patch[i*8+3], patch[i*8+4], patch[i*8+5], patch[i*8+6], patch[i*8+7]], dim=2)
            whole = torch.cat([whole, whole_part], dim=1)
        return whole