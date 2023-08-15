import torch
import torch.nn as nn
import torch.nn.functional as F

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class SEBasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, net_type='se_net', reduction=16):
        super(SEBasicBlock, self).__init__()
        self.se = SELayer(out_channels, reduction)
        self.downsample = downsample
        self.stride = stride
        self.se_layer = True if net_type == 'se_net' else False

        kernels = [[3,1],[1,3],[1,1]]

        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = kernels[0], stride = [stride,1], padding = [1,0]),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())

        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = kernels[1], stride = [1,stride], padding = [0,1]),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())

        self.conv3 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = kernels[2], stride = [1,1], padding = 0),
                        nn.BatchNorm2d(out_channels))

        self.relu = nn.ReLU()

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        if self.se_layer:
            out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out
    
class EmbeddingNetwork(nn.Module):
    def __init__(self, cfg, block):
        super(EmbeddingNetwork, self).__init__()
        self.inplanes = 16
        self.out_shape = cfg['embed_dim']
        self.layers = cfg['SE_blocks']
        self.conv = nn.Sequential(
                        nn.Conv2d(4, 16, kernel_size = [12,2], stride = [3,1], padding = [11,0]),
                        nn.BatchNorm2d(16),
                        nn.ReLU())
        self.senet1 = self._make_layer(block, 16, self.layers[0], 'se_net', 2)
        self.resnet1 = self._make_layer(block, 64, self.layers[1], 'resnet', 2)
        self.senet2 = self._make_layer(block, 64, self.layers[2], 'se_net', 2)
        self.resnet2 = self._make_layer(block, 128, self.layers[3], 'resnet', 2)

        self.tdd1 = nn.Sequential(
                        nn.Conv2d(1, 64, (256, 1)),
                        nn.ReLU(),
        )
        self.tdd2 = nn.Sequential(
                        nn.Conv2d(1, self.out_shape, (64, 1)),
                        nn.ReLU(),
        )

    def _make_layer(self, block, planes, blocks, net_type, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, downsample=downsample, net_type=net_type))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, net_type=net_type))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        x = self.senet1(x)
        x = self.resnet1(x)
        x = self.senet2(x)
        x = self.resnet2(x)
        
        x = x.reshape(x.shape[0],1,-1,x.shape[-1])
        x = self.tdd1(x)
        x = x.reshape(x.shape[0],1,-1,x.shape[-1])
        x = self.tdd2(x)
        x = x.reshape(x.shape[0],self.out_shape,-1)

        return x