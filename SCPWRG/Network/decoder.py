from .patch_encoder import *
from .who_encoder import *
from .unet_parts import *

class Decoder(nn.Module):
    def __init__(self, channel, basic_channel=16, output_channel=1):
        super(Decoder, self).__init__()

        self.channel = channel
        self.basic_channel = basic_channel

        self.up_1 = Up_block(basic_channel*16, basic_channel*4)
        self.up_2 = Up_block(basic_channel*8, basic_channel*2)
        self.up_3 = Up_block(basic_channel*4, basic_channel*1)
        self.up_4 = Up_block(basic_channel*2, basic_channel)

        self.fin_conv_1 = block(basic_channel, 3)
        self.fin_conv_2 = block(3, output_channel)
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, d0, d1, d2, d3, d4):
        u1 = self.up_1(d4, d3)
        u2 = self.up_2(u1, d2)
        u3 = self.up_3(u2, d1)
        u4 = self.up_4(u3, d0)

        x = self.fin_conv_1(u4)
        x = self.fin_conv_2(x)

        return self.sigmoid(x)