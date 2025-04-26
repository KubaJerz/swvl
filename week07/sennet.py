import torch.nn as nn

#same bacuise we go 3x3, 3x3, 3x3
class BasicBlockSame(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride, bias=False)
        self.bnorm0 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(out_channels)

        self.shortcut  = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        residual = x
        x = self.conv0(x)
        x = self.bnorm0(x)
        x = nn.functional.relu(x)

        x = self.conv1(x)
        x = self.bnorm1(x)
        x = nn.functional.relu(x)

        x = self.conv2(x)
        x = self.bnorm2(x)

        x += self.shortcut(residual)
        x = nn.functional.relu(x)
        return x


#varying bacuise we go 3x3, 1x1, 3x3
class BasicBlockVarying(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1 ):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, out_channels, 3, padding=1, stride=stride, bias=False)
        self.bnorm0 = nn.BatchNorm2d(out_channels)

        self.conv1 = nn.Conv2d(out_channels, out_channels, 1, bias=False)
        self.bnorm1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.bnorm2 = nn.BatchNorm2d(out_channels)

        self.shortcut  = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Conv2d(in_channels, out_channels, 1, stride=stride)

    def forward(self, x):
        residual = x
        x = self.conv0(x)
        x = self.bnorm0(x)
        x = nn.functional.relu(x)

        x = self.conv1(x)
        x = self.bnorm1(x)
        x = nn.functional.relu(x)

        x = self.conv2(x)
        x = self.bnorm2(x)



        x += self.shortcut(residual)
        x = nn.functional.relu(x)
        return x


class SENBlock(nn.Module):
    def __init__(self, in_channels, r=16):
        super().__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.fc0 = nn.Linear(in_channels, int(in_channels/r))
        self.fc1 = nn.Linear(int(in_channels/r), in_channels)

    def forward(self, x):
        residual = x
        batch_size, channels, _, _ = residual.size()
        x = self.sq(x).squeeze()
        x = self.fc0(x)
        x = nn.functional.relu(x)

        x = self.fc1(x)
        x = nn.functional.sigmoid(x).view(batch_size, channels, 1, 1)
        
        x = residual * x
        return x
    
class SenResNetBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.c0 = nn.Conv2d(3, 32, 3, padding=1)
        self.maxp = nn.MaxPool2d(2, 2)
        self.resblock0 = BasicBlockSame(32, 64, stride=2)
        self.resblock1 = BasicBlockSame(64, 64)
        self.resblock2 = BasicBlockSame(64, 64)
        self.sen0 = SENBlock(64)

        self.resblock3 = BasicBlockVarying(64, 128, stride=2)
        self.resblock4 = BasicBlockVarying(128, 128)
        self.resblock5 = BasicBlockVarying(128, 128)
        self.resblock6 = BasicBlockVarying(128, 128)
        self.sen1 = SENBlock(128)

        self.resblock7 = BasicBlockVarying(128, 256, stride=2)
        self.resblock8 = BasicBlockVarying(256, 256)
        self.resblock9 = BasicBlockVarying(256, 256)
        self.resblock10 = BasicBlockVarying(256, 256)
        self.resblock11 = BasicBlockVarying(256, 256)
        self.resblock12 = BasicBlockVarying(256, 256)
        self.sen2 = SENBlock(256)

        self.resblock13 = BasicBlockVarying(256, 512, stride=2)
        self.resblock14 = BasicBlockVarying(512, 512)
        self.resblock15 = BasicBlockVarying(512, 512)

        self.c1 = nn.Conv2d(512, 1024, 3, padding=1, stride=2)
        self.c2 = nn.Conv2d(1024, 1024, 1, padding=0) 

    def forward(self, x):
        x = self.c0(x)
        x = self.maxp(x)

        x = self.resblock0(x)
        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.sen0(x)

        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        x = self.sen1(x)

        x = self.resblock7(x)
        x = self.resblock8(x)
        x = self.resblock9(x)
        x = self.resblock10(x)
        x = self.resblock11(x)
        x = self.resblock12(x)
        x = self.sen2(x)

        x = self.resblock13(x)
        x = self.resblock14(x)
        x = self.resblock15(x)

        x = self.c1(x)
        x = self.c2(x)
        
        return x 

class ClassificationHead(nn.Module):
    def __init__(self, in_features=1024, num_classes=1000):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(in_features, num_classes)
    
    def forward(self, x):
        x = self.gap(x).squeeze()
        x = self.fc(x)
        return x

class SenResNet(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        self.backbone = SenResNetBackbone()
        self.classifier = ClassificationHead(in_features=1024, num_classes=num_classes)
    
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output
    
    def get_backbone(self):
        return self.backbone
    
    def get_feature_dim(self):
        return 1024