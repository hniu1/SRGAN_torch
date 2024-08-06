import torch
import torch.nn as nn
import torch.nn.functional as F


class ResidualBlock(nn.Module):
    def __init__(self):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        # self.bn1 = nn.BatchNorm2d(num_features=64)
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        # self.bn2 = nn.BatchNorm2d(num_features=64)

        # Initialize weights
        nn.init.trunc_normal_(self.conv1.weight, std=0.02)
        nn.init.trunc_normal_(self.conv2.weight, std=0.02)
        # nn.init.trunc_normal_(self.bn1.weight, mean=1.0, std=0.02)
        # nn.init.trunc_normal_(self.bn2.weight, mean=1.0, std=0.02)

    def forward(self, x):
        z = F.relu(self.conv1(x))
        z = self.conv2(z)
        x = x + z
        return x
    
class SRGAN_g(nn.Module):
    def __init__(self):
        super(SRGAN_g, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        self.subpixel_conv1 = nn.PixelShuffle(upscale_factor=2)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=576, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        self.subpixel_conv2 = nn.PixelShuffle(upscale_factor=3)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.conv5 = nn.Conv2d(
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
        x = self.relu(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = x + temp
        x = self.subpixel_conv1(x)
        x = self.conv3(x)
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.subpixel_conv2(x)
        x = self.conv4(x)
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.conv6(x)
        x = self.conv5(x)
        return x
    
class SRGAN_g_lr(nn.Module):
    def __init__(self):
        super(SRGAN_g_lr, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=64, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.relu = nn.ReLU(inplace=True)
        self.residual_block = self.make_layer()
        self.conv2 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        self.subpixel_conv1 = nn.PixelShuffle(upscale_factor=2)
        self.conv3 = nn.Conv2d(
            in_channels=16, out_channels=256, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        self.subpixel_conv2 = nn.PixelShuffle(upscale_factor=2)
        self.conv4 = nn.Conv2d(
            in_channels=64, out_channels=64, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        self.conv6 = nn.Conv2d(
            in_channels=64, out_channels=32, kernel_size=(3, 3), stride=(1, 1), padding=1, bias=False
        )
        self.conv5 = nn.Conv2d(
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
        x = self.relu(self.conv1(x))
        temp = x
        x = self.residual_block(x)
        x = self.conv2(x)
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = x + temp
        x = self.subpixel_conv1(x)
        x = self.conv3(x)
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.subpixel_conv2(x)
        x = self.conv4(x)
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.conv6(x)
        x = self.conv5(x)
        return x

class SRGAN_d(nn.Module):
    def __init__(self, hr_size, dim=64):
        super(SRGAN_d, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=1, out_channels=dim, kernel_size=(3, 3), stride=(1, 1), padding=1
        )
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=dim)
        self.conv3 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=dim)
        self.conv4 = nn.Conv2d(
            in_channels=dim, out_channels=dim * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(num_features=dim * 2)
        self.conv5 = nn.Conv2d(
            in_channels=dim * 2, out_channels=dim * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False
        )
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
        x = self.lrelu(self.conv1(x))
        x = self.conv2(x)
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.lrelu(self.bn1(x))
        x = self.conv3(x)
        x = self.lrelu(self.bn2(x))
        x = self.conv4(x)
        x = self.lrelu(self.bn3(x))
        x = self.conv5(x)
        x = self.lrelu(self.bn4(x))
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
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=(4, 4), stride=(1, 1), padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(num_features=dim)
        self.conv3 = nn.Conv2d(
            in_channels=dim, out_channels=dim, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(num_features=dim)
        self.conv4 = nn.Conv2d(
            in_channels=dim, out_channels=dim * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(num_features=dim * 2)
        self.conv5 = nn.Conv2d(
            in_channels=dim * 2, out_channels=dim * 2, kernel_size=(4, 4), stride=(2, 2), padding=1, bias=False
        )
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
        x = self.lrelu(self.conv1(x))
        x = self.conv2(x)
        x= F.pad(x, (0, 1, 1, 0))  # [left, right, top, bot]
        x = self.lrelu(self.bn1(x))
        x = self.conv3(x)
        x = self.lrelu(self.bn2(x))
        x = self.conv4(x)
        x = self.lrelu(self.bn3(x))
        x = self.conv5(x)
        x = self.lrelu(self.bn4(x))
        x = self.flat(x)
        x = self.dense(x)
        # x = torch.sigmoid(x)  # Apply sigmoid function after the dense layer
        return x