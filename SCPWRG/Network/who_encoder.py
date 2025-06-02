from .unet_parts import *
import torch.nn as nn

class WEncoder(nn.Module):
    def __init__(self, n_channel, basic_channel=16):
        super(WEncoder, self).__init__()
        self.n_channels = n_channel
        self.basic_channel = basic_channel

        self.db_0 = block(self.n_channels, basic_channel)
        self.db_1 = down_block(basic_channel, basic_channel*2)
        self.db_2 = down_block(basic_channel*2, basic_channel*4)
        self.db_3 = down_block(basic_channel*4, basic_channel*8)
        self.db_4 = down_block(basic_channel*8, basic_channel*16)

    def forward(self, x):
        d0 = self.db_0(x)
        d1 = self.db_1(d0)
        d2 = self.db_2(d1)
        d3 = self.db_3(d2)
        d4 = self.db_4(d3)

        return d0, d1, d2, d3, d4
