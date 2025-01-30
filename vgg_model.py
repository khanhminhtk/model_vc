from torchsummary import summary
import torch.nn as nn
import torch

class Cnn_block(nn.Module):
    def __init__(self, out_channels, in_channels = 3, padding = 1):
        super(Cnn_block, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = 3, padding = padding),
            nn.BatchNorm2d(out_channels),
            nn.GELU(),
        )
    def forward(self, x):
        return self.conv(x)
    
class Fully_connected(nn.Module):
    def __init__(self, in_features, out_features, dropout = 0.5):
        super(Fully_connected, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=out_features),
            nn.Dropout(dropout),
            nn.GELU(),
        )
    def forward(self, x):
        return self.fc(x)

class VGG19_etraction(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.block_1 = nn.Sequential(
            Cnn_block(in_channels=3, out_channels=64, padding=1),
            Cnn_block(in_channels=64, out_channels=64, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )# (3, 224, 224) -> (64, 112, 112)
        self.block_2 = nn.Sequential(
            Cnn_block(in_channels=64, out_channels=128, padding=1),
            Cnn_block(in_channels=128, out_channels=128, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )# (64, 112, 112) -> (128, 56, 56)
        self.block_3 = nn.Sequential(
            Cnn_block(in_channels=128, out_channels=256, padding=1),
            Cnn_block(in_channels=256, out_channels=256, padding=1),
            Cnn_block(in_channels=256, out_channels=256, padding=1),
            Cnn_block(in_channels=256, out_channels=256, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )# (128, 56, 56) -> (256, 28, 28)
        self.block_4 = nn.Sequential(
            Cnn_block(in_channels=256, out_channels=512, padding=1),
            Cnn_block(in_channels=512, out_channels=512, padding=1),
            Cnn_block(in_channels=512, out_channels=512, padding=1),
            Cnn_block(in_channels=512, out_channels=512, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )# (256, 28, 28) -> (512, 14, 14)
        self.block_5 = nn.Sequential(
            Cnn_block(in_channels=512, out_channels=512, padding=1),
            Cnn_block(in_channels=512, out_channels=512, padding=1),
            Cnn_block(in_channels=512, out_channels=512, padding=1),
            Cnn_block(in_channels=512, out_channels=512, padding=1),   
            nn.MaxPool2d(kernel_size=2, stride=2)
        )# (512, 14, 14) -> (512, 7, 7)
    def forward(self, x):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.block_4(x)
        x = self.block_5(x)
        return x

class VGG19(nn.Module):
    def __init__(self, num_classes):
        super(VGG19, self).__init__()
        self.extractor = VGG19_etraction()
        self.classifier = nn.Sequential(
            Fully_connected(in_features=512*7*7, out_features=4096),
            Fully_connected(in_features=4096, out_features=1000),
            nn.Linear(in_features=1000, out_features=num_classes)
        )

    def forward(self, x):
        x = self.extractor(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
    
if __name__ == '__main__':
    model = VGG19(num_classes=10)
    print(summary(model, (3, 224, 224), device='cpu'))