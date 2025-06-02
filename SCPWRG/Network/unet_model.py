""" Full assembly of the parts to form the complete network """

from .unet_parts import *
class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        basic_channel = 32

        self.db_0 = block(self.n_channels, basic_channel)
        self.db_1 = down_block(basic_channel, basic_channel*2)
        self.db_2 = down_block(basic_channel*2, basic_channel*4)
        self.db_3 = down_block(basic_channel*4, basic_channel*8)
        self.db_4 = down_block(basic_channel*8, basic_channel*16)

        self.mid_conv = nn.Conv2d(basic_channel*16, basic_channel*8,kernel_size=3, padding=1)

        self.up_1 = Up_block(basic_channel*16, basic_channel*4)
        self.up_2 = Up_block(basic_channel*8, basic_channel*2)
        self.up_3 = Up_block(basic_channel*4, basic_channel*1)
        self.up_4 = Up_block(basic_channel*2, basic_channel)
        
        self.fin_conv_1 = block(basic_channel, 3)
        self.fin_conv_2 = block(3, 1)
        self.sigmoid = torch.nn.Sigmoid()

        
    def forward(self, x):
        d0 = self.db_0(x)
        d1 = self.db_1(d0)
        d2 = self.db_2(d1)
        d3 = self.db_3(d2)
        d4 = self.db_4(d3)

        d4 = self.mid_conv(d4)
        
        u1 = self.up_1(d4, d3)
        u2 = self.up_2(u1, d2)
        u3 = self.up_3(u2, d1)
        u4 = self.up_4(u3, d0)

        x = self.fin_conv_1(u4)
        x = self.fin_conv_2(x)
        

        return self.sigmoid(x)

class block(nn.Module):
    def __init__(self, in_channel, output_channel):
        super().__init__()

        self.res_conv = nn.Conv2d(in_channel, output_channel, kernel_size=3, padding=1)

        self.block = nn.Sequential(
        nn.Conv2d(in_channel, output_channel, kernel_size=3, padding=1),
        nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1),
        nn.BatchNorm2d(output_channel),
        nn.ReLU(),
        )
    def forward(self, x):
        return self.res_conv(x) + self.block(x)

class down_block(nn.Module):
    def __init__(self, in_channel, output_channel):
        super().__init__()
        self.down = nn.MaxPool2d(2)
        self.block = block(in_channel, output_channel)
    def forward(self, x):
        x = self.down(x)
        x = self.block(x)
        return x

class Up_block(nn.Module):
    def __init__(self, in_channel, output_channel):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.block = block(in_channel, output_channel)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x2, x1], dim=1)

        return self.block(x)