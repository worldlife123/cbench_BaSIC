import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super(UNetDown, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, 4, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super(UNetUp, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x


class GeneratorUNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, 
                 mid_channels=[64, 128, 256, 512, 512, 512, 512],
                 dropout_probs=[0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 0.5],):
        super(GeneratorUNet, self).__init__()
        self.mid_channels = mid_channels

        self.input_layer = UNetDown(in_channels, mid_channels[0], dropout=dropout_probs[0], normalize=False)
        
        self.down_layers = nn.ModuleList()
        self.up_layers = nn.ModuleList()
        for i in range(1, len(mid_channels)):
            self.down_layers.append(UNetDown(mid_channels[i-1], mid_channels[i], dropout=dropout_probs[i]))
            self.up_layers.append(UNetUp(mid_channels[i]*2, mid_channels[i-1], dropout=dropout_probs[i]))
        self.down_layers.append(UNetDown(mid_channels[-1], mid_channels[-1], normalize=False, dropout=dropout_probs[-1]))
        self.up_layers.append(UNetUp(mid_channels[-1], mid_channels[-1], dropout=dropout_probs[-1]))
        # self.down1 = UNetDown(in_channels, 64, normalize=False)
        # self.down2 = UNetDown(64, 128)
        # self.down3 = UNetDown(128, 256)
        # self.down4 = UNetDown(256, 512, dropout=0.5)
        # self.down5 = UNetDown(512, 512, dropout=0.5)
        # self.down6 = UNetDown(512, 512, dropout=0.5)
        # self.down7 = UNetDown(512, 512, dropout=0.5)
        # self.down8 = UNetDown(512, 512, normalize=False, dropout=0.5)

        # self.up1 = UNetUp(512, 512, dropout=0.5)
        # self.up2 = UNetUp(1024, 512, dropout=0.5)
        # self.up3 = UNetUp(1024, 512, dropout=0.5)
        # self.up4 = UNetUp(1024, 512, dropout=0.5)
        # self.up5 = UNetUp(1024, 256)
        # self.up6 = UNetUp(512, 128)
        # self.up7 = UNetUp(256, 64)

        self.final_layer = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(mid_channels[0]*2, out_channels, 4, padding=1),
            # nn.Tanh(),
        )

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        # d1 = self.down1(x)
        # d2 = self.down2(d1)
        # d3 = self.down3(d2)
        # d4 = self.down4(d3)
        # d5 = self.down5(d4)
        # d6 = self.down6(d5)
        # d7 = self.down7(d6)
        # d8 = self.down8(d7)
        # u1 = self.up1(d8, d7)
        # u2 = self.up2(u1, d6)
        # u3 = self.up3(u2, d5)
        # u4 = self.up4(u3, d4)
        # u5 = self.up5(u4, d3)
        # u6 = self.up6(u5, d2)
        # u7 = self.up7(u6, d1)
        x = self.input_layer(x)
        down_features = [x]
        for layer in self.down_layers:
            down_features.append(layer(x))
            x = down_features[-1]

        for i in range(len(self.up_layers)-1, -1, -1):
            x = self.up_layers[i](x, down_features[i])

        return self.final_layer(x)
