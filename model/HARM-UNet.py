import numpy as np
import torch
from torch import nn
from einops import rearrange
import torch.nn.functional as F

from model.MSFEM import MSFEM
from model.DSSA import DSSA
from model.PCFN import PCFN
from model.SFGM import SFGM
from model.DFEM import DFEM

class ConvBNReLU(nn.Module):

    def __init__(self, c_in, c_out, kernel_size,
                 stride=1, padding=1, activation=True):

        super(ConvBNReLU, self).__init__()

        self.conv = nn.Conv2d(
            c_in, c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.bn = nn.BatchNorm2d(c_out)
        self.relu = nn.ReLU()
        self.activation = activation

    def forward(self, x):

        x = self.conv(x)
        x = self.bn(x)

        if self.activation:
            x = self.relu(x)

        return x

class DoubleConv(nn.Module):

    def __init__(self, cin, cout):

        super(DoubleConv, self).__init__()

        self.conv = nn.Sequential(
            ConvBNReLU(cin, cout, 3, 1, padding=1),
            ConvBNReLU(cout, cout, 3,
                       stride=1,
                       padding=1,
                       activation=False)
        )

        self.conv1 = nn.Conv2d(cout, cout, 1)
        self.bn = nn.BatchNorm2d(cout)
        self.relu = nn.ReLU()

    def forward(self, x):

        x = self.conv(x)
        h = x
        x = self.conv1(x)
        x = self.bn(x)
        x = h + x
        x = self.relu(x)

        return x

class U_encoder(nn.Module):

    def __init__(self):

        super(U_encoder, self).__init__()

        self.res1 = MSFEM(3, 64)
        self.pool1 = nn.MaxPool2d(2)

        self.res2 = MSFEM(64, 128)
        self.pool2 = nn.MaxPool2d(2)

        self.res3 = MSFEM(128, 256)
        self.pool3 = nn.MaxPool2d(2)

    def forward(self, x):

        features = []

        x = self.res1(x)
        features.append(x)
        x = self.pool1(x)

        x = self.res2(x)
        features.append(x)
        x = self.pool2(x)

        x = self.res3(x)
        features.append(x)
        x = self.pool3(x)

        return x, features

class U_decoder(nn.Module):

    def __init__(self):

        super(U_decoder, self).__init__()

        self.sfgm0 = SFGM(64, 64, 64)
        self.sfgm1 = SFGM(128, 128, 128)
        self.sfgm2 = SFGM(256, 256, 256)

        self.trans1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.trans2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.trans3 = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.res1 = DoubleConv(512, 256)
        self.res2 = DoubleConv(256, 128)
        self.res3 = DoubleConv(128, 64)

    def forward(self, x, feature):

        Ps = []

        x = self.trans1(x)
        feature[2] = self.sfgm2(feature[2], x)
        x = torch.cat((feature[2], x), dim=1)
        x = self.res1(x)
        P3 = x
        Ps.append(P3)

        x = self.trans2(x)
        feature[1] = self.sfgm1(feature[1], x)
        x = torch.cat((feature[1], x), dim=1)
        x = self.res2(x)
        P4 = x
        Ps.append(P4)

        x = self.trans3(x)
        feature[0] = self.sfgm0(feature[0], x)
        x = torch.cat((feature[0], x), dim=1)
        x = self.res3(x)
        P5 = x
        Ps.append(P5)

        return x, Ps

class SPIB(nn.Module):

    def __init__(self, dim):

        super(SPIB, self).__init__()

        self.SlayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.CSAttention = DSSA(dim)

        self.ElayerNorm = nn.LayerNorm(dim, eps=1e-6)
        self.pcfn = PCFN(dim)

    def forward(self, x):

        h = x
        x = self.SlayerNorm(x)
        x = self.CSAttention(x)
        x = h + x

        h = x
        x = self.ElayerNorm(x)
        x = self.pcfn(x)
        x = h + x

        return x

class Stem(nn.Module):

    def __init__(self):

        super(Stem, self).__init__()

        self.model = U_encoder()
        self.trans_dim = ConvBNReLU(256, 256, 1, 1, 0)
        self.position_embedding = nn.Parameter(torch.zeros((1, 784, 256)))

    def forward(self, x):

        x, features = self.model(x)
        x = self.trans_dim(x)
        x = x.flatten(2)
        x = x.transpose(-2, -1)
        x = x + self.position_embedding

        return x, features

class DecoderStem(nn.Module):

    def __init__(self):

        super(DecoderStem, self).__init__()
        self.block = U_decoder()

    def forward(self, x, features):

        x, Ps = self.block(x, features)

        return x, Ps

class encoder_block(nn.Module):

    def __init__(self, dim):

        super(encoder_block, self).__init__()

        self.block = nn.ModuleList([
            SPIB(dim),
            ConvBNReLU(dim, dim * 2, 2, stride=2, padding=0)
        ])

    def forward(self, x):

        x = self.block[0](x)
        B, N, C = x.shape
        h, w = int(np.sqrt(N)), int(np.sqrt(N))
        x = x.view(B, h, w, C).permute(0, 3, 1, 2)
        skip = x
        x = self.block[1](x)

        return x, skip

class decoder_block(nn.Module):

    def __init__(self, dim, flag):

        super(decoder_block, self).__init__()
        self.flag = flag
        self.dfem = DFEM(dim // 2, dim // 2)

        if not self.flag:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(
                    dim,
                    dim // 2,
                    kernel_size=2,
                    stride=2,
                    padding=0
                ),
                nn.Conv2d(
                    dim,
                    dim // 2,
                    kernel_size=1,
                    stride=1
                ),
                SPIB(dim // 2)
            ])

        else:
            self.block = nn.ModuleList([
                nn.ConvTranspose2d(
                    dim,
                    dim // 2,
                    kernel_size=2,
                    stride=2,
                    padding=0
                ),
                SPIB(dim)
            ])

    def forward(self, x, skip):

        if not self.flag:
            x = self.block[0](x)
            skip = self.dfem(skip)
            x = torch.cat((x, skip), dim=1)
            x = self.block[1](x)
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.block[2](x)
            O = x

        else:
            x = self.block[0](x)
            skip = self.dfem(skip)
            x = torch.cat((x, skip), dim=1)
            x = x.permute(0, 2, 3, 1)
            B, H, W, C = x.shape
            x = x.view(B, -1, C)
            x = self.block[1](x)
            O = x

        return x, O

class HARMUNet(nn.Module):

    def __init__(self, out_ch=4):

        super(HARMUNet, self).__init__()
        self.stem = Stem()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()
        self.decoder_stem = DecoderStem()
        for dim in configs["encoder"]:
            self.encoder.append(encoder_block(dim))

        for dim in configs["decoder"][:-1]:
            self.decoder.append(decoder_block(dim, False))
        self.decoder.append(
            decoder_block(configs["decoder"][-1], True)
        )
        self.ds_heads = nn.ModuleList([
            nn.Conv2d(512, out_ch, 1),
            nn.Conv2d(512, out_ch, 1),
            nn.Conv2d(256, out_ch, 1),
            nn.Conv2d(128, out_ch, 1),
            nn.Conv2d(64, out_ch, 1),
        ])

    def forward(self, x):

        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        x, features = self.stem(x)
        skips = []

        for encoder in self.encoder:
            x, skip = encoder(x)
            skips.append(skip)
            B, C, H, W = x.shape
            x = x.permute(0, 2, 3, 1).contiguous().view(B, -1, C)

        B, N, C = x.shape

        x = x.view(B,
                   int(np.sqrt(N)),
                   int(np.sqrt(N)),
                   C).permute(0, 3, 1, 2)
        Os = []

        for i, decoder in enumerate(self.decoder):
            x, O = decoder(x, skips[len(self.decoder) - i - 1])
            B, N, C = x.shape
            B1, N1, C1 = O.shape
            x = x.view(B,
                       int(np.sqrt(N)),
                       int(np.sqrt(N)),
                       C).permute(0, 3, 1, 2)
            O = O.view(B1,
                       int(np.sqrt(N1)),
                       int(np.sqrt(N1)),
                       C1).permute(0, 3, 1, 2)
            Os.append(O)

        x, Ps = self.decoder_stem(x, features)
        Os = Os + Ps

        assert len(Os) == len(self.ds_heads), \
            "The number of ds_heads does not match the number of features."
        ds_outputs = []

        for feat, head in zip(Os, self.ds_heads):
            out = head(feat)
            out = F.interpolate(out,
                                size=(224, 224),
                                mode='bilinear',
                                align_corners=False)
            ds_outputs.append(out)

        return ds_outputs

configs = {
    "encoder": [256, 512],
    "bottleneck": 1024,
    "decoder": [1024, 512],
}
