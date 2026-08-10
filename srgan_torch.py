import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=64)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=64)
        # self.dropout = nn.Dropout(p=0.05)  # Adjust the dropout rate as needed

        # Initialize weights
        nn.init.trunc_normal_(self.conv1.weight, std=0.02)
        nn.init.trunc_normal_(self.conv2.weight, std=0.02)
        # nn.init.trunc_normal_(self.bn1.weight, mean=0.0, std=0.02)
        # nn.init.trunc_normal_(self.bn2.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # z = F.relu(self.conv1(x))
        z = self.prelu(self.bn1(self.conv1(x)))
        # z = self.dropout(z)
        z = self.bn2(self.conv2(z))
        x = x + z
        return x

class ResidualBlock_256(nn.Module):
    def __init__(self):
        super(ResidualBlock_256, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=256)
        self.prelu = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=256)
        # self.dropout = nn.Dropout(p=0.05)  # Adjust the dropout rate as needed

        # Initialize weights
        nn.init.trunc_normal_(self.conv1.weight, std=0.02)
        nn.init.trunc_normal_(self.conv2.weight, std=0.02)
        # nn.init.trunc_normal_(self.bn1.weight, mean=0.0, std=0.02)
        # nn.init.trunc_normal_(self.bn2.weight, mean=0.0, std=0.02)

    def forward(self, x):
        # z = F.relu(self.conv1(x))
        z = self.prelu(self.bn1(self.conv1(x)))
        # z = self.dropout(z)
        z = self.bn2(self.conv2(z))
        x = x + z
        return x
    
class SRGAN_g(nn.Module):
    def __init__(self):
        super(SRGAN_g, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu1 = nn.PReLU()
        # self.relu = nn.ReLU(inplace=False)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        self.prelu2 = nn.PReLU()
        self.subpixel_conv1 = nn.PixelShuffle(upscale_factor=2)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=576, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        self.prelu3 = nn.PReLU()
        self.subpixel_conv2 = nn.PixelShuffle(upscale_factor=3)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def make_layer(self):
        layers = []
        for _ in range(16):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # x = self.relu(self.conv1(x))
        x = self.prelu1(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.prelu2(self.conv2(x))
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = x + temp
        x = self.subpixel_conv1(x)
        x = self.prelu3(self.conv3(x))
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.subpixel_conv2(x)
        x = self.prelu4(self.conv4(x))
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.prelu5(self.conv5(x))
        x = self.conv6(x)
        return x
    
class SRGAN_g_lr(nn.Module):
    def __init__(self):
        super(SRGAN_g_lr, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu1 = nn.PReLU()
        self.relu = nn.ReLU(inplace=False)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        self.prelu2 = nn.PReLU()
        self.subpixel_conv1 = nn.PixelShuffle(upscale_factor=2)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=256, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        self.prelu3 = nn.PReLU()
        self.subpixel_conv2 = nn.PixelShuffle(upscale_factor=2)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def make_layer(self):
        layers = []
        for _ in range(16):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        # x = self.relu(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.prelu2(self.conv2(x))
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = x + temp
        x = self.subpixel_conv1(x)
        x = self.prelu3(self.conv3(x))
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.subpixel_conv2(x)
        x = self.prelu4(self.conv4(x))
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.prelu5(self.conv5(x))
        x = self.conv6(x)
        return x

class SRGAN_g_lr_smallFeature(nn.Module):
    def __init__(self, in_channels):
        super(SRGAN_g_lr_smallFeature, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu1 = nn.PReLU()
        self.relu = nn.ReLU(inplace=False)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same', bias=False
        )
        self.prelu2 = nn.PReLU()
        self.subpixel_conv1 = nn.PixelShuffle(upscale_factor=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=128, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        # Optionally add a smoothing convolution here
        self.smoothing_conv = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu3 = nn.PReLU()
        self.subpixel_conv2 = nn.PixelShuffle(upscale_factor=2)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def make_layer(self):
        layers = []
        for _ in range(16):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        # x = self.relu(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        # x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = x + temp
        x = self.smoothing_conv(x)
        x = self.subpixel_conv1(x)
        x = self.conv3(x)
        # x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        # x = self.smoothing_conv(x)
        x = self.subpixel_conv2(x)
        x = self.prelu4(self.conv4(x))
        # x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.conv5(x)
        x = self.conv6(x)
        return x
    
class SRGAN_g_lr_20(nn.Module):
    def __init__(self, in_channels):
        super(SRGAN_g_lr_20, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu1 = nn.PReLU()
        self.relu = nn.ReLU(inplace=False)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(5, 5), stride=(1, 1), padding='same', bias=False
        )
        self.prelu2 = nn.PReLU()
        self.subpixel_conv1 = nn.PixelShuffle(upscale_factor=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        # Optionally add a smoothing convolution here
        self.smoothing_conv = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu3 = nn.PReLU()
        self.subpixel_conv2 = nn.PixelShuffle(upscale_factor=2)
        self.conv4 = nn.Conv2d(
            in_channels=32, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def make_layer(self):
        layers = []
        for _ in range(16):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = x + temp
        x = self.subpixel_conv1(x)
        x = self.conv3(x)
        x = self.smoothing_conv(x)
        x = self.subpixel_conv2(x)
        x = self.prelu4(self.conv4(x))
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class SRGAN_g_lr_25(nn.Module):
    def __init__(self, in_channels):
        super(SRGAN_g_lr_25, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu1 = nn.PReLU()
        self.relu = nn.ReLU(inplace=False)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu2 = nn.PReLU()
        self.subpixel_conv1 = nn.PixelShuffle(upscale_factor=2)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        # Optionally add a smoothing convolution here
        self.smoothing_conv = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu3 = nn.PReLU()
        self.subpixel_conv2 = nn.PixelShuffle(upscale_factor=2)
        self.conv4 = nn.Conv2d(
            in_channels=16, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def make_layer(self):
        layers = []
        for _ in range(16):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = x + temp
        x = self.subpixel_conv1(x)
        x = self.conv3(x)
        x = self.smoothing_conv(x)
        x = self.subpixel_conv2(x)
        x = self.prelu4(self.conv4(x))
        x = self.conv5(x)
        x = self.conv6(x)
        return x
    
class SRGAN_g_lr_26(nn.Module):
    def __init__(self, in_channels):
        super(SRGAN_g_lr_26, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu1 = nn.PReLU()
        self.relu = nn.ReLU(inplace=False)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu2 = nn.PReLU()
        self.subpixel_conv1 = nn.PixelShuffle(upscale_factor=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        # Optionally add a smoothing convolution here
        self.smoothing_conv = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu3 = nn.PReLU()
        self.subpixel_conv2 = nn.PixelShuffle(upscale_factor=2)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def make_layer(self):
        layers = []
        for _ in range(16):
            layers.append(ResidualBlock_256())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = x + temp
        x = self.subpixel_conv1(x)
        x = self.conv3(x)
        x = self.smoothing_conv(x)
        x = self.subpixel_conv2(x)
        x = self.prelu4(self.conv4(x))
        x = self.conv5(x)
        x = self.conv6(x)
        return x


class PatchResidualBlock(nn.Module):
    """Small-patch residual block without zero padding or BatchNorm."""

    def __init__(self, channels=64):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
        )
        self.act = nn.PReLU(channels)
        self.conv2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
        )

    def forward(self, x):
        return x + self.conv2(self.act(self.conv1(x)))


class SRGAN_g_lr_patch(nn.Module):
    """Generator designed for 8x8 LR patches and 4x super-resolution.

    The original 16-block generator has a receptive field much larger than an
    8x8 patch and repeatedly introduces zeros at every patch edge.  This model
    is deliberately shallow, uses reflection padding, omits BatchNorm, keeps
    PixelShuffle upsampling, and learns a correction to a bilinear baseline.
    """

    def __init__(self, in_channels, channels=64, num_residual_blocks=1):
        super().__init__()
        if in_channels < 1:
            raise ValueError("in_channels must include the temperature channel")

        self.stem = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=0, bias=True),
            nn.PReLU(channels),
        )
        self.residual_blocks = nn.Sequential(
            *[PatchResidualBlock(channels) for _ in range(num_residual_blocks)]
        )
        self.trunk = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, kernel_size=3, padding=0, bias=True),
        )
        self.up1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels * 4, kernel_size=3, padding=0, bias=True),
            nn.PixelShuffle(2),
            nn.PReLU(channels),
        )
        self.up2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels * 4, kernel_size=3, padding=0, bias=True),
            nn.PixelShuffle(2),
            nn.PReLU(channels),
        )
        self.output = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, 1, kernel_size=3, padding=0, bias=True),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_in")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Start from bilinear interpolation; training learns only corrections.
        nn.init.zeros_(self.output[1].weight)
        nn.init.zeros_(self.output[1].bias)

    def forward(self, x):
        baseline = F.interpolate(
            x[:, :1], scale_factor=4, mode="bilinear", align_corners=False
        )
        features = self.stem(x)
        features = features + self.trunk(self.residual_blocks(features))
        features = self.up1(features)
        correction = self.output(self.up2(features))
        return baseline + correction


class SRGAN_g_lr_patch_hr_elev(nn.Module):
    """Deeper patch generator conditioned on native 0.25-degree elevation.

    LR input channels are temperature and 1-degree elevation.  The separate
    HR elevation tensor is fused only after the two PixelShuffle stages.  Its
    anomaly relative to upscaled LR elevation explicitly describes unresolved
    sub-grid terrain.
    """

    def __init__(self, in_channels=2, channels=96, num_residual_blocks=4):
        super().__init__()
        if in_channels < 2:
            raise ValueError("Expected LR temperature and elevation channels")
        self.stem = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, channels, 3, padding=0, bias=True),
            nn.PReLU(channels),
        )
        self.residual_blocks = nn.Sequential(
            *[PatchResidualBlock(channels) for _ in range(num_residual_blocks)]
        )
        self.trunk = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels, 3, padding=0, bias=True),
        )
        self.up1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels * 4, 3, padding=0, bias=True),
            nn.PixelShuffle(2),
            nn.PReLU(channels),
        )
        self.up2 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels * 4, 3, padding=0, bias=True),
            nn.PixelShuffle(2),
            nn.PReLU(channels),
        )
        # HR features + HR elevation + upscaled LR elevation + anomaly.
        self.hr_fusion = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels + 3, channels, 3, padding=0, bias=True),
            nn.PReLU(channels),
            PatchResidualBlock(channels),
            PatchResidualBlock(channels),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels, channels // 2, 3, padding=0, bias=True),
            nn.PReLU(channels // 2),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels // 2, 1, 3, padding=0, bias=True),
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, a=0.2, mode="fan_in")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Begin from bilinear temperature; learn terrain-aware corrections.
        final = self.hr_fusion[-1]
        nn.init.zeros_(final.weight)
        nn.init.zeros_(final.bias)

    def forward(self, x, elevation_hr):
        if elevation_hr.ndim == 3:
            elevation_hr = elevation_hr[:, None]
        baseline = F.interpolate(
            x[:, :1], scale_factor=4, mode="bilinear", align_corners=False
        )
        elevation_lr_hr = F.interpolate(
            x[:, 1:2], scale_factor=4, mode="bilinear", align_corners=False
        )
        if elevation_hr.shape[-2:] != baseline.shape[-2:]:
            raise ValueError(
                f"HR elevation {elevation_hr.shape[-2:]} does not match "
                f"4x output {baseline.shape[-2:]}"
            )
        features = self.stem(x)
        features = features + self.trunk(self.residual_blocks(features))
        features = self.up2(self.up1(features))
        terrain_anomaly = elevation_hr - elevation_lr_hr
        correction = self.hr_fusion(torch.cat(
            (features, elevation_hr, elevation_lr_hr, terrain_anomaly), dim=1
        ))
        return baseline + correction

class SRGAN_g_hr_26(nn.Module):
    def __init__(self, in_channels):
        super(SRGAN_g_hr_26, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu1 = nn.PReLU()
        self.relu = nn.ReLU(inplace=False)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu2 = nn.PReLU()
        self.subpixel_conv1 = nn.PixelShuffle(upscale_factor=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=576, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        # Optionally add a smoothing convolution here
        self.smoothing_conv = nn.Conv2d(
            in_channels=576, out_channels=576, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu3 = nn.PReLU()
        self.subpixel_conv2 = nn.PixelShuffle(upscale_factor=3)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def make_layer(self):
        layers = []
        for _ in range(16):
            layers.append(ResidualBlock_256())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = x + temp
        x = self.subpixel_conv1(x)
        x = self.conv3(x)
        x = self.smoothing_conv(x)
        x = self.subpixel_conv2(x)
        x = self.prelu4(self.conv4(x))
        x = self.conv5(x)
        x = self.conv6(x)
        return x
    
class SRGAN_g_hr_26_64RB(nn.Module):
    def __init__(self, in_channels):
        super(SRGAN_g_hr_26_64RB, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu1 = nn.PReLU()
        self.relu = nn.ReLU(inplace=False)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu2 = nn.PReLU()
        self.conv31 = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.subpixel_conv1 = nn.PixelShuffle(upscale_factor=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=576, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
            # in_channels=64, out_channels=576, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        # Optionally add a smoothing convolution here
        self.smoothing_conv = nn.Conv2d(
            in_channels=576, out_channels=576, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
            # in_channels=576, out_channels=576, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu3 = nn.PReLU()
        self.subpixel_conv2 = nn.PixelShuffle(upscale_factor=3)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def make_layer(self):
        layers = []
        for _ in range(16):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = x + temp
        x = self.conv31(x)
        x = self.subpixel_conv1(x)
        x = self.conv3(x)
        x = self.smoothing_conv(x)
        x = self.subpixel_conv2(x)
        x = self.prelu4(self.conv4(x))
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class SRGAN_g_hhr_64RB(nn.Module): # from 100km to 4km
    def __init__(self, in_channels):
        super(SRGAN_g_hhr_64RB, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu1 = nn.PReLU()
        self.relu = nn.ReLU(inplace=False)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu2 = nn.PReLU()
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.subpixel_conv1 = nn.PixelShuffle(upscale_factor=2)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=576, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        # Optionally add a smoothing convolution here
        self.smoothing_conv1 = nn.Conv2d(
            in_channels=576, out_channels=576, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu4 = nn.PReLU()
        self.subpixel_conv2 = nn.PixelShuffle(upscale_factor=3)
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.smoothing_conv2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu5 = nn.PReLU()
        self.subpixel_conv3 = nn.PixelShuffle(upscale_factor=2)
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.smoothing_conv3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.subpixel_conv4 = nn.PixelShuffle(upscale_factor=2)
        self.conv7 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.conv8 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu8 = nn.PReLU()
        self.conv9 = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def make_layer(self):
        layers = []
        for _ in range(16):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = x + temp
        x = self.conv3(x)
        x = self.subpixel_conv1(x)
        x = self.conv4(x)
        x = self.smoothing_conv1(x)
        x = self.subpixel_conv2(x)
        x = self.prelu5(self.conv5(x))
        x = self.smoothing_conv2(x)
        x = self.subpixel_conv3(x)
        x = self.conv6(x)
        x = self.smoothing_conv3(x)
        x = self.subpixel_conv4(x)
        x = self.conv7(x)
        x = self.prelu8(self.conv8(x))
        x = self.conv9(x)
        return x


class SRGAN_g_hr_26_64RB(nn.Module):
    def __init__(self, in_channels):
        super(SRGAN_g_hr_26_64RB, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu1 = nn.PReLU()
        self.relu = nn.ReLU(inplace=False)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu2 = nn.PReLU()
        self.conv31 = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.subpixel_conv1 = nn.PixelShuffle(upscale_factor=2)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        # Optionally add a smoothing convolution here
        self.smoothing_conv = nn.Conv2d(
            in_channels=256, out_channels=576, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu3 = nn.PReLU()
        self.subpixel_conv2 = nn.PixelShuffle(upscale_factor=3)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def make_layer(self):
        layers = []
        for _ in range(8):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = x + temp
        x = self.conv31(x)
        x = self.subpixel_conv1(x)
        x = self.conv3(x)
        x = self.smoothing_conv(x)
        x = self.subpixel_conv2(x)
        x = self.prelu4(self.conv4(x))
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class SRGAN_g_lr_31(nn.Module):
    def __init__(self, in_channels):
        super(SRGAN_g_lr_31, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu1 = nn.PReLU()
        self.relu = nn.ReLU(inplace=False)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu2 = nn.PReLU()
        self.up1 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        # Optionally add a smoothing convolution here
        self.prelu3 = nn.PReLU()
        self.up2 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def make_layer(self):
        layers = []
        for _ in range(16):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        # x = self.relu(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = x + temp
        x = self.up1(x)
        x = self.conv3(x)
        x = self.up2(x)
        x = self.prelu4(self.conv4(x))
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class SRGAN_g_lr_32(nn.Module):
    def __init__(self, in_channels):
        super(SRGAN_g_lr_32, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu1 = nn.PReLU()
        self.relu = nn.ReLU(inplace=False)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu2 = nn.PReLU()
        self.up1 = nn.ConvTranspose2d(64, 64, 4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.smooth = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        # Optionally add a smoothing convolution here
        self.prelu3 = nn.PReLU()
        self.up2 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def make_layer(self):
        layers = []
        for _ in range(16):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        # x = self.relu(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = x + temp
        x = self.up1(x)
        x = self.conv3(x)
        x = self.smooth(x)
        x = self.up2(x)
        x = self.prelu4(self.conv4(x))
        x = self.conv5(x)
        x = self.conv6(x)
        return x
    
class SRGAN_g_lr_33(nn.Module):
    def __init__(self, in_channels):
        super(SRGAN_g_lr_33, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=in_channels, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu1 = nn.PReLU()
        self.relu = nn.ReLU(inplace=False)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu2 = nn.PReLU()
        self.up1 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
        self.conv3 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.smooth = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        # Optionally add a smoothing convolution here
        self.prelu3 = nn.PReLU()
        self.up2 = nn.ConvTranspose2d(256, 256, 4, stride=2, padding=1, bias=False)
        self.conv4 = nn.Conv2d(
            in_channels=256, out_channels=256, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(
            in_channels=256, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def make_layer(self):
        layers = []
        for _ in range(16):
            layers.append(ResidualBlock_256())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        # x = self.relu(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x = x + temp
        x = self.up1(x)
        x = self.conv3(x)
        x = self.smooth(x)
        x = self.up2(x)
        x = self.prelu4(self.conv4(x))
        x = self.conv5(x)
        x = self.conv6(x)
        return x

class SRGAN_g_lr_upsample(nn.Module):
    def __init__(self):
        super(SRGAN_g_lr_upsample, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu1 = nn.PReLU()
        self.relu = nn.ReLU(inplace=False)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu2 = nn.PReLU()
        # self.subpixel_conv1 = nn.PixelShuffle(upscale_factor=2)
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv3 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu3 = nn.PReLU()
        # self.subpixel_conv2 = nn.PixelShuffle(upscale_factor=2)
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu4 = nn.PReLU()
        self.conv5 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu5 = nn.PReLU()
        self.conv6 = nn.Conv2d(
            in_channels=32, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=0, bias=False
        )

        # Initialize weights
        self._initialize_weights()

    def make_layer(self):
        layers = []
        for _ in range(16):
            layers.append(ResidualBlock())
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        # x = self.relu(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.prelu2(self.conv2(x))
        # x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = x + temp
        # x = self.subpixel_conv1(x)
        x = self.upsample1(x)
        x = self.prelu3(self.conv3(x))
        # x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        # x = self.subpixel_conv2(x)
        x = self.upsample2(x)
        x = self.prelu4(self.conv4(x))
        # x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.prelu5(self.conv5(x))
        x = self.conv6(x)
        return x

class SRGAN_d(nn.Module):
    def __init__(self, hr_size, dim=64):
        super(SRGAN_d, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=dim, kernel_size=(3, 3), stride=(1, 1), padding=1
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        # self.prelu2 = nn.PReLU()
        self.bn1 = nn.BatchNorm2d(num_features=dim)
        self.conv3 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False
        )
        # self.prelu3 = nn.PReLU()
        self.bn2 = nn.BatchNorm2d(num_features=dim)
        self.conv4 = nn.Conv2d(
            in_channels=dim, out_channels=dim * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False
        )
        # self.prelu4 = nn.PReLU()
        self.bn3 = nn.BatchNorm2d(num_features=dim * 2)
        self.conv5 = nn.Conv2d(
            in_channels=dim * 2, out_channels=dim * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False
        )
        # self.prelu5 = nn.PReLU()
        self.bn4 = nn.BatchNorm2d(num_features=dim * 2)
        self.flat = nn.Flatten()
        self.dense = nn.Linear(in_features=hr_size*2,out_features=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.conv2(x)
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.conv5(x)
        x = self.bn4(x)
        x = self.flat(x)
        x = self.dense(x)
        # x = torch.sigmoid(x)  # Apply sigmoid function after the dense layer
        return x
    
class SRGAN_d_lr(nn.Module):
    def __init__(self, hr_size, dim=64):
        super(SRGAN_d_lr, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=dim, kernel_size=(3, 3), stride=(1, 1), padding=1
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        self.prelu2 = nn.PReLU()
        self.bn1 = nn.BatchNorm2d(num_features=dim)
        self.conv3 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False
        )
        self.prelu3 = nn.PReLU()
        self.bn2 = nn.BatchNorm2d(num_features=dim)
        self.conv4 = nn.Conv2d(
            in_channels=dim, out_channels=dim * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False
        )
        self.prelu4 = nn.PReLU()
        self.bn3 = nn.BatchNorm2d(num_features=dim * 2)
        self.conv5 = nn.Conv2d(
            in_channels=dim * 2, out_channels=dim * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False
        )
        self.prelu5 = nn.PReLU()
        self.bn4 = nn.BatchNorm2d(num_features=dim * 2)
        self.flat = nn.Flatten()
        self.dense = nn.Linear(in_features=53760,out_features=1)
        # self.dense = nn.Linear(in_features=53760,out_features=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.conv2(x)
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.prelu2(self.bn1(x))
        x = self.conv3(x)
        x = self.prelu3(self.bn2(x))
        x = self.conv4(x)
        x = self.prelu4(self.bn3(x))
        x = self.conv5(x)
        x = self.prelu4(self.bn4(x))
        x = self.flat(x)
        x = self.dense(x)
        # x = torch.sigmoid(x)  # Apply sigmoid function after the dense layer
        return x
    
class SRGAN_d_lr_odd(nn.Module):
    def __init__(self, hr_size, dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, dim, 3, 1, 1)
        # self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.prelu1 = nn.PReLU()
        self.prelu2 = nn.PReLU()
        self.prelu3 = nn.PReLU()
        self.prelu4 = nn.PReLU()
        self.prelu5 = nn.PReLU()
        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        # self.bn1 = nn.BatchNorm2d(dim)
        self.conv3 = nn.Conv2d(dim, dim, 5, 2, 2, bias=False)
        # self.bn2 = nn.BatchNorm2d(dim)
        self.conv4 = nn.Conv2d(dim, dim * 2, 5, 2, 2, bias=False)
        # self.bn3 = nn.BatchNorm2d(dim * 2)
        self.conv5 = nn.Conv2d(dim * 2, dim * 2, 5, 2, 2, bias=False)
        # self.bn4 = nn.BatchNorm2d(dim * 2)
        self.flat = nn.Flatten()

        # --- auto-compute dense layer size ---
        with torch.no_grad():
            hr_h, hr_w = hr_size
            dummy = torch.zeros(1, 1, hr_h, hr_w) # N, C, H, W
            out = self._forward_features(dummy)
            num_features = out.view(1, -1).size(1)
        self.dense = nn.Linear(num_features, 1)

        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.trunc_normal_(m.weight, mean=1.0, std=0.02)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def _forward_features(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.conv2(x)
        x = self.prelu2(x)
        x = self.conv3(x)
        x = self.prelu3(x)
        x = self.conv4(x)
        x = self.prelu4(x)
        x = self.conv5(x)
        x = self.prelu5(x)
        return x

    def forward(self, x):
        x = self._forward_features(x)
        x = self.flat(x)
        x = self.dense(x)
        return x
    
# class SRGAN_d_lr_odd(nn.Module):
#     def __init__(self, hr_size, dim=64):
#         super(SRGAN_d_lr_odd, self).__init__()
#         self.conv1 = nn.Conv2d(
#             in_channels=1, out_channels=dim, kernel_size=(3, 3), stride=(1, 1), padding='same'
#         )
#         self.lrelu = nn.LeakyReLU(0.2, inplace=False)
#         self.prelu1 = nn.PReLU()
#         self.conv2 = nn.Conv2d(
#             in_channels=dim, out_channels=dim, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
#         )
#         self.prelu2 = nn.PReLU()
#         self.bn1 = nn.BatchNorm2d(num_features=dim)
#         self.conv3 = nn.Conv2d(
#             in_channels=dim, out_channels=dim, kernel_size=(5, 5), stride=(2, 2), padding=2, bias=False
#         )
#         self.prelu3 = nn.PReLU()
#         self.bn2 = nn.BatchNorm2d(num_features=dim)
#         self.conv4 = nn.Conv2d(
#             in_channels=dim, out_channels=dim * 2, kernel_size=(5, 5), stride=(2, 2), padding=2, bias=False
#         )
#         self.prelu4 = nn.PReLU()
#         self.bn3 = nn.BatchNorm2d(num_features=dim * 2)
#         self.conv5 = nn.Conv2d(
#             in_channels=dim * 2, out_channels=dim * 2, kernel_size=(5, 5), stride=(2, 2), padding=2, bias=False
#         )
#         self.prelu5 = nn.PReLU()
#         self.bn4 = nn.BatchNorm2d(num_features=dim * 2)
#         self.flat = nn.Flatten()
#         self.dense = nn.Linear(in_features=57600,out_features=1)

#         # Initialize weights
#         self._initialize_weights()

#     def _initialize_weights(self):
#         for m in self.modules():
#             if isinstance(m, nn.Conv2d):
#                 nn.init.trunc_normal_(m.weight, std=0.02)
#                 if m.bias is not None:
#                     nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.BatchNorm2d):
#                 nn.init.trunc_normal_(m.weight, mean=1.0, std=0.02)
#                 nn.init.constant_(m.bias, 0)
#             elif isinstance(m, nn.Linear):
#                 nn.init.trunc_normal_(m.weight, std=0.02)
#                 nn.init.constant_(m.bias, 0)

#     def forward(self, x):
#         x = self.prelu1(self.conv1(x))
#         x = self.conv2(x)
#         # x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
#         x = self.prelu2(self.bn1(x))
#         x = self.conv3(x)
#         x = self.prelu3(self.bn2(x))
#         x = self.conv4(x)
#         x = self.prelu4(self.bn3(x))
#         x = self.conv5(x)
#         x = self.prelu4(self.bn4(x))
#         x = self.flat(x)
#         x = self.dense(x)
#         # x = torch.sigmoid(x)  # Apply sigmoid function after the dense layer
#         return x

class SRGAN_d_hr_gap(nn.Module):
    """
    HR discriminator with global average pooling.
    Resolution-agnostic, NCCL-safe, and memory efficient.
    """
    def __init__(self, dim=64):
        super().__init__()

        # -------- Convolutional backbone --------
        self.conv1 = nn.Conv2d(1, dim, 3, 1, 1)
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(dim, dim, 3, 1, 1, bias=False)
        # self.bn1 = nn.BatchNorm2d(dim)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(dim, dim, 5, 2, 2, bias=False)
        # self.bn2 = nn.BatchNorm2d(dim)
        self.prelu3 = nn.PReLU()

        self.conv4 = nn.Conv2d(dim, dim * 2, 5, 2, 2, bias=False)
        # self.bn3 = nn.BatchNorm2d(dim * 2)
        self.prelu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(dim * 2, dim * 2, 5, 2, 2, bias=False)
        # self.bn4 = nn.BatchNorm2d(dim * 2)
        self.prelu5 = nn.PReLU()

        # -------- Global pooling + classifier --------
        self.gap = nn.AdaptiveAvgPool2d(1)   # (N, C, 1, 1)
        self.dense = nn.Linear(dim * 2, 1)

        self._initialize_weights()
    # -------------------------------------------------
    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.conv2(x))
        x = self.prelu3(self.conv3(x))
        x = self.prelu4(self.conv4(x))
        x = self.prelu5(self.conv5(x))

        x = self.gap(x)          # (N, C, 1, 1)
        x = x.view(x.size(0), -1)  # (N, C)
        x = self.dense(x)
        return x
    # -------------------------------------------------
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            # elif isinstance(m, nn.BatchNorm2d):
            #     nn.init.trunc_normal_(m.weight, mean=1.0, std=0.02)
            #     nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)
    
class SRGAN_d_hr_odd(nn.Module):
    def __init__(self, hr_size, dim=64):
        super(SRGAN_d_hr_odd, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=dim, kernel_size=(3, 3), stride=(1, 1), padding=1
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.prelu2 = nn.PReLU()
        self.bn1 = nn.BatchNorm2d(num_features=dim)
        self.conv3 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=(5, 5), stride=(2, 2), padding=2, bias=False
        )
        self.prelu3 = nn.PReLU()
        self.bn2 = nn.BatchNorm2d(num_features=dim)
        self.conv4 = nn.Conv2d(
            in_channels=dim, out_channels=dim * 2, kernel_size=(5, 5), stride=(2, 2), padding=2, bias=False
        )
        self.prelu4 = nn.PReLU()
        self.bn3 = nn.BatchNorm2d(num_features=dim * 2)
        self.conv5 = nn.Conv2d(
            in_channels=dim * 2, out_channels=dim * 2, kernel_size=(5, 5), stride=(2, 2), padding=2, bias=False
        )
        self.prelu5 = nn.PReLU()
        self.bn4 = nn.BatchNorm2d(num_features=dim * 2)
        self.flat = nn.Flatten()
        self.dense = nn.Linear(in_features=1971072,out_features=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.conv2(x)
        # x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.prelu2(self.bn1(x))
        x = self.conv3(x)
        x = self.prelu3(self.bn2(x))
        x = self.conv4(x)
        x = self.prelu4(self.bn3(x))
        x = self.conv5(x)
        x = self.prelu4(self.bn4(x))
        x = self.flat(x)
        x = self.dense(x)
        # x = torch.sigmoid(x)  # Apply sigmoid function after the dense layer
        return x
    
class SRGAN_d_hr(nn.Module):
    def __init__(self, hr_size, dim=64):
        super(SRGAN_d_hr, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=dim, kernel_size=(5, 5), stride=(1, 1), padding=1
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=(5, 5), stride=(1, 1), padding=1, bias=False
        )
        self.prelu2 = nn.PReLU()
        self.bn1 = nn.BatchNorm2d(num_features=dim)
        self.conv3 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=(9, 9), stride=(2, 2), padding=1, bias=False
        )
        self.prelu3 = nn.PReLU()
        self.bn2 = nn.BatchNorm2d(num_features=dim)
        self.conv4 = nn.Conv2d(
            in_channels=dim, out_channels=dim * 2, kernel_size=(9, 9), stride=(2, 2), padding=1, bias=False
        )
        self.prelu4 = nn.PReLU()
        self.bn3 = nn.BatchNorm2d(num_features=dim * 2)
        self.conv5 = nn.Conv2d(
            in_channels=dim * 2, out_channels=dim * 2, kernel_size=(5, 5), stride=(1, 1), padding=1, bias=False
        )
        self.prelu5 = nn.PReLU()
        self.bn4 = nn.BatchNorm2d(num_features=dim * 2)
        self.flat = nn.Flatten()
        self.dense = nn.Linear(in_features=7417472,out_features=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.conv2(x)
        # x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.prelu2(self.bn1(x))
        x = self.conv3(x)
        x = self.prelu3(self.bn2(x))
        x = self.conv4(x)
        x = self.prelu4(self.bn3(x))
        x = self.conv5(x)
        x = self.prelu4(self.bn4(x))
        x = self.flat(x)
        x = self.dense(x)
        # x = torch.sigmoid(x)  # Apply sigmoid function after the dense layer
        return x

class SRGAN_d_lr_large(nn.Module):
    def __init__(self, hr_size, dim=64):
        super(SRGAN_d_lr_large, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=dim, kernel_size=(3, 3), stride=(1, 1), padding='same'
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=False)
        self.prelu1 = nn.PReLU()
        self.conv2 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=(5, 5), stride=(1, 1), padding='same', bias=False
        )
        self.prelu2 = nn.PReLU()
        self.bn1 = nn.BatchNorm2d(num_features=dim)
        self.conv3 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False
        )
        self.prelu3 = nn.PReLU()
        self.bn2 = nn.BatchNorm2d(num_features=dim)
        self.conv4 = nn.Conv2d(
            in_channels=dim, out_channels=dim * 2, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False
        )
        self.prelu4 = nn.PReLU()
        self.bn3 = nn.BatchNorm2d(num_features=dim * 2)
        self.conv5 = nn.Conv2d(
            in_channels=dim * 2, out_channels=dim * 2, kernel_size=(5, 5), stride=(2, 2), padding=2, bias=False
        )
        self.prelu5 = nn.PReLU()
        self.bn4 = nn.BatchNorm2d(num_features=dim * 2)
        self.conv6 = nn.Conv2d(
            in_channels=dim * 2, out_channels=dim * 2, kernel_size=(5, 5), stride=(1, 1), padding='same', bias=False
        )
        self.prelu6 = nn.PReLU()
        self.bn5 = nn.BatchNorm2d(num_features=dim * 2)
        self.conv7 = nn.Conv2d(
            in_channels=dim * 2, out_channels=dim * 2, kernel_size=(5, 5), stride=(1, 1), padding='same', bias=False
        )
        self.prelu7 = nn.PReLU()
        self.bn6 = nn.BatchNorm2d(num_features=dim * 2)
        self.conv8 = nn.Conv2d(
            in_channels=dim * 2, out_channels=dim * 2, kernel_size=(3, 3), stride=(1, 1), padding='same', bias=False
        )
        self.prelu8 = nn.PReLU()
        self.bn7 = nn.BatchNorm2d(num_features=dim * 2)

        self.flat = nn.Flatten()
        self.dense = nn.Linear(in_features=57600,out_features=1)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.trunc_normal_(m.weight, mean=1.0, std=0.02)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.conv2(x)
        # x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.bn1(x)
        x = self.conv3(x)
        x = self.bn2(x)
        x = self.conv4(x)
        x = self.bn3(x)
        x = self.conv5(x)
        x = self.bn4(x)
        temp = x
        x = self.conv6(x)
        x = self.bn5(x)
        x = self.conv7(x)
        x = self.bn6(x)
        x = self.conv8(x)
        x = self.bn7(x)
        x += temp
        x = self.flat(x)
        x = self.dense(x)
        # x = torch.sigmoid(x)  # Apply sigmoid function after the dense layer
        return x


class SRGAN_d_lr_odd_3h(nn.Module):
    def __init__(self, hr_size, dim=64):
        super().__init__()
        self.conv1 = nn.Conv2d(1, dim, 3, 1, padding='same')
        self.prelu1 = nn.PReLU()

        self.conv2 = nn.Conv2d(dim, dim, 3, 1, padding='same', bias=False)
        self.bn1   = nn.GroupNorm(32, dim)          # or BatchNorm2d(dim)
        self.prelu2 = nn.PReLU()

        self.conv3 = nn.Conv2d(dim, dim, 5, 2, padding=2, bias=False)
        self.bn2   = nn.GroupNorm(32, dim)
        self.prelu3 = nn.PReLU()

        self.conv4 = nn.Conv2d(dim, dim*2, 5, 2, padding=2, bias=False)
        self.bn3   = nn.GroupNorm(32, dim*2)
        self.prelu4 = nn.PReLU()

        self.conv5 = nn.Conv2d(dim*2, dim*2, 5, 2, padding=2, bias=False)
        self.bn4   = nn.GroupNorm(32, dim*2)
        self.prelu5 = nn.PReLU()

        # robust head (no hardcoded 57600)
        self.gap   = nn.AdaptiveAvgPool2d(1)
        self.flat  = nn.Flatten()
        self.dense = nn.Linear(dim*2, 1)

        # init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02); nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.prelu1(self.conv1(x))
        x = self.prelu2(self.bn1(self.conv2(x)))
        x = self.prelu3(self.bn2(self.conv3(x)))
        x = self.prelu4(self.bn3(self.conv4(x)))
        x = self.prelu5(self.bn4(self.conv5(x)))     # use prelu5 here
        x = self.gap(x)
        x = self.flat(x)
        x = self.dense(x)
        return x
