import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init

class WRNBasicblock(nn.Module):
    def __init__(self, in_planes, out_planes, stride):
        super(WRNBasicblock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.equalInOut = (in_planes == out_planes)
        self.convShortcut = (not self.equalInOut) and nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, padding=0, bias=False) or None

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(x))
        if not self.equalInOut: residual = out  
        out = self.conv2(self.relu(self.bn2(self.conv1(out))))
        if self.convShortcut is not None: residual = self.convShortcut(residual)
        return out + residual

class WRN(nn.Module):
    def __init__(self, depth, wide, num_classes):
        super(WRN, self).__init__()

        n_channels = [16, 16*wide, 32*wide, 64*wide]
        assert((depth - 4) % 6 == 0)
        layer_blocks = int((depth - 4) / 6)
        print ('WRN : Depth : {} , Widen Factor : {}'.format(depth, wide))

        self.num_classes = num_classes
        self.conv_3x3 = nn.Conv2d(3, n_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        
        self.stage_1 = self._make_layer(n_channels[0], n_channels[1], layer_blocks, 1)
        self.stage_2 = self._make_layer(n_channels[1], n_channels[2], layer_blocks, 2)
        self.stage_3 = self._make_layer(n_channels[2], n_channels[3], layer_blocks, 2)
        self.lastact = nn.Sequential(nn.BatchNorm2d(n_channels[3]), nn.ReLU(inplace=True))
        self.avgpool = nn.AvgPool2d(8)
        self.classifier = nn.Linear(n_channels[3], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight)
                m.bias.data.zero_()

    def _make_layer(self, in_planes, out_planes, num_layers, stride):
        layers = []
        layers.append(WRNBasicblock(in_planes, out_planes, stride))
        for i in range(1, num_layers):
            layers.append(WRNBasicblock(out_planes, out_planes, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv_3x3(x)
        x = self.stage_1(x)
        x = self.stage_2(x)
        x = self.stage_3(x)
        x = self.lastact(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)

def wrn(depth, wide, num_classes):
    model = WRN(depth, wide, num_classes)
    return model
