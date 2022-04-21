import torch
from torch import nn

class ResidualConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1, padding=1):
        super().__init__()

        self.conv_block = nn.Sequential(
          nn.BatchNorm2d(in_ch),
          nn.ReLU(),
          nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
          nn.BatchNorm2d(out_ch),
          nn.ReLU(),
          nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
          )
        self.conv_skip = nn.Sequential(
          nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=1),
          nn.BatchNorm2d(out_ch),
        )
      
    def forward(self, x):
        return self.conv_block(x) + self.conv_skip(x)

class UpSampleConvs(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.relu  = nn.ReLU()
        self.upSample = nn.Upsample(scale_factor=2)
        self.INorm = torch.nn.InstanceNorm2d(out_ch)

    def forward(self, x):
        x = self.upSample(x)
        x = self.conv(x)
        x = self.relu(x)
        x = self.INorm(x)
        return x
    
class ResUnet(nn.Module):
    def __init__(self, num_classes=1,
                 channel=3,
                 filters=[64, 128, 256, 512],
                 activation=nn.Sigmoid()):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )
        self.residual_conv_1 = ResidualConvBlock(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ResidualConvBlock(filters[1], filters[2], 2, 1)
        self.bridge = ResidualConvBlock(filters[2], filters[3], 2, 1)
        self.upsample_1 = UpSampleConvs(filters[3], filters[3])
        self.up_residual_conv1 = ResidualConvBlock(filters[3] + filters[2], filters[2], 1, 1)
        self.upsample_2 = UpSampleConvs(filters[2], filters[2])
        self.up_residual_conv2 = ResidualConvBlock(filters[2] + filters[1], filters[1], 1, 1)
        self.upsample_3 = UpSampleConvs(filters[1], filters[1])
        self.up_residual_conv3 = ResidualConvBlock(filters[1] + filters[0], filters[0], 1, 1)
#         self.output_layer = nn.Sequential(
#             nn.Conv2d(filters[0], num_classes, 1, 1),
#             activation,
#         )

    def forward(self, x):
        # Encode
        EncFtrs = []
        x = self.input_layer(x) + self.input_skip(x)
        EncFtrs.append(x)
        x = self.residual_conv_1(x)
        EncFtrs.append(x)
        x = self.residual_conv_2(x)
        EncFtrs.append(x)

        # Bridge
        x = self.bridge(EncFtrs[2])
        # Decode
        x = self.upsample_1(x)
        x = torch.cat([x, EncFtrs[2]], dim=1)
        x = self.up_residual_conv1(x)
        x = self.upsample_2(x)
        x = torch.cat([x, EncFtrs[1]], dim=1)
        x = self.up_residual_conv2(x)
        x = self.upsample_3(x)
        x = torch.cat([x, EncFtrs[0]], dim=1)
        x = self.up_residual_conv3(x)
        return x
#         output = self.output_layer(x)
#         return output