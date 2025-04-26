import torch
import torch.nn as nn
import torchvision.models as models

class RegHead(nn.Module):
    def __init__(self,in_channels, S, B, dropout=0.0):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels, 1024, 3, padding=1)
        self.conv1 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv2 = nn.Conv2d(1024, 1024, 3, padding=1)
        self.conv3 = nn.Conv2d(1024, 1024, 3, padding=1)

        # self.l0 = nn.Linear((4096 * S * S), 4096)
        self.l0 = nn.Linear((1024 * S * S), 4096)
        self.drop = nn.Dropout(dropout)
        self.act = nn.LeakyReLU(0.1)
        self.l1 = nn.Linear(4096, (S * S * (B * 5)))

    def forward(self, x):
        x = self.conv0(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x.flatten(start_dim=1)
        x = self.l0(x)
        x = self.drop(x)
        x = self.act(x)
        x = self.l1(x)
        return x




class FaceDetector(nn.Module):
    def __init__(self, S=7, B=2, dropout=0.0):
        super(FaceDetector, self).__init__()

        backbone = models.resnet50(pretrained=True)
        self.backbone = torch.nn.Sequential(*(list(backbone.children())[:-2]))
        
        self.regression_head = RegHead(2048, S, B, dropout=dropout)
    
    def forward(self, x):
        features = self.backbone(x)
        out = self.regression_head(features)
        return out