import torch.nn as nn
from torchsummary import summary
import io
from contextlib import redirect_stdout

class CnnBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, padding, stride = 1):
        super(CnnBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels = in_channels,
                out_channels = out_channels,
                kernel_size = kernel_size,
                stride = stride,
                padding = padding,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.conv(x)

class BottleneckBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, padding = 1, reduction_factor = 1, expansion_factor = 1):
        super(BottleneckBlock, self).__init__()
        in_intermediate = int(in_channels * expansion_factor / reduction_factor) #128 #128 128
        self.botteleneck = nn.Sequential(
            CnnBlock(in_channels = in_channels, # [[64, 256, 512
                    out_channels = in_intermediate, # [[128, 128
                    kernel_size = 1,
                    stride = 1,
                    padding = 0 
                    ), # [[(128, 56, 56) (128, 56, 56)
            CnnBlock(in_channels = in_intermediate, #[[128, 128
                    out_channels = in_intermediate, #[[128, 128
                    kernel_size = 3, 
                    stride = stride,
                    padding = padding
                    ), # [[(128, 56, 56) (128, 56, 56)
            CnnBlock(in_channels = in_intermediate, #[[128, 128
                    out_channels = out_channels, #[[256, 256
                    kernel_size = 1,
                    stride = 1,
                    padding = 0
                    ) # [[(256, 56, 56), (256, 56, 56)
            )
        self.identity = nn.Sequential(
            CnnBlock(
                in_channels = in_channels, # [[64, 256
                out_channels = out_channels, #[[256, 256]]
                kernel_size = 1,
                stride = stride,
                padding = 0,
            )
        ) # [[(256, 56, 56)
    def forward(self, x):
        return self.botteleneck(x) + self.identity(x) #[[(256, 56, 56), 

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.layer_1 = CnnBlock(
            in_channels = 3,
            out_channels = 64,
            kernel_size = 7,
            padding = 3,
            stride = 2
        ) # (3, 224, 224) => (64, 112, 112)
        self.layer_2 = nn.Sequential(
            nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1), # (64, 112, 112) => (64, 56, 56)
            self._make_layer(num_blocks = 3, in_channels = 64, out_channels = 256, stride = 1)
        )
        self.layer_3 = nn.Sequential(
            self._make_layer(num_blocks = 8, in_channels = 256, out_channels = 512, stride = 2)
        )
        self.layer_4 = nn.Sequential(
            self._make_layer(num_blocks = 36, in_channels = 512, out_channels = 1024, stride = 2)
        )
        self.layer_5 = nn.Sequential(
            self._make_layer(num_blocks = 3, in_channels = 1024, out_channels = 2048, stride = 2)
        )
        self.average_pool = nn.AvgPool2d(kernel_size = 7)
    def _make_layer(self, num_blocks, in_channels, out_channels, stride):
        layer = []
        if stride == 1:
            temp_reduction_factor, temp_expansion_factor = 1, 2 
        else:
            temp_reduction_factor, temp_expansion_factor = 2, 1 

        layer.append(
            BottleneckBlock(
                in_channels=in_channels,
                out_channels=out_channels,
                stride=stride,
                reduction_factor=temp_reduction_factor,
                expansion_factor=temp_expansion_factor
            )
        )
        in_channels = out_channels
                                                         
        for _ in range(1, num_blocks):
            layer.append(
                BottleneckBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    stride=1,  
                    reduction_factor=2 * stride,  
                    expansion_factor=1
                )
            )
        return nn.Sequential(*layer)

    def forward(self, x):
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.layer_5(x)
        x = self.average_pool(x)
        return x

class ResNet152_classification(nn.Module):
    def __init__(self, num_class):
        super(ResNet152_classification, self).__init__()
        self.backbone = ResNet()
        self.MLP = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = 2048, out_features = 1000),
            nn.Linear(in_features = 1000, out_features = num_class)
        )

    def forward(self, x):
        x = self.backbone(x)
        x = self.MLP(x)
        return x

if __name__ == "__main__":
    model = ResNet152_classification(num_class=10)
    print(summary(model, (3, 224, 224), device='cpu'))
    print(model)
    with open("./resnet_architecture.txt", "w") as file:
        file.write(str(model))
    with open("./resnet_summary.txt", "w") as file:
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            summary(model, (3, 224, 224), device='cpu')
        file.write(buffer.getvalue())
