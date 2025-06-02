from .patch_encoder import *
from .who_encoder import *
from .unet_parts import *
from .decoder import *

class Model(nn.Module):
    def __init__(self, w_channel, p_channel, basic_channel=16):
        super(Model, self).__init__()

        self.w_channel = w_channel
        self.p_channel = p_channel
        self.basic_channel = basic_channel

        self.w_encoder = WEncoder(self.w_channel, self.basic_channel)
        self.p_encoder = PEncoder(self.p_channel, self.basic_channel)

        self.w_decoder = Decoder(self.w_channel, self.basic_channel, 1)
        self.p_decoder = Decoder(self.p_channel, self.basic_channel, 1)
        self.multi_decoder = Decoder(self.w_channel, self.basic_channel, 1)

        self.mix_conv = block(basic_channel*32, basic_channel*8)
        self.middle_conv = block(basic_channel*16, basic_channel*8)


    def forward(self, x, x_patch, adj):
        d0_w, d1_w, d2_w, d3_w, d4_w = self.w_encoder(x)

        d0_p, d1_p, d2_p, d3_p, d4_p, whole = self.p_encoder(x_patch, adj)

        d4_w = torch.cat([d4_w, whole], dim=1)
        d4_w = self.mix_conv(d4_w)

        d4_p = self.middle_conv(d4_p)

        o_w = self.w_decoder(d0_w, d1_w, d2_w, d3_w, d4_w)
        o_multi = self.multi_decoder(d0_w, d1_w, d2_w, d3_w, d4_w)
        o_p = self.p_decoder(d0_p, d1_p, d2_p, d3_p, d4_p)

        return o_w, o_p, o_multi