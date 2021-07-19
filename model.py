import torch
from torch._C import CONV_BN_FUSION
from torch.functional import split
import torch.nn as nn
from torch.nn.modules import padding

architecture_config = [
    # tuple = (kernel size, number of filters of output, stride, padding)
    (7, 64, 2, 3),  # 1번째 layer 그림
    "M",
    (3, 192, 1, 1), # 2번째 그림
    "M",
    (1, 128, 1, 0), # 3번째
    (3, 256, 1, 1),
    (1, 256, 1, 0),
    (3, 512, 1, 1),
    "M",
    [(1, 256, 1, 0), (3, 512, 1, 1), 4],    # 4번째
    #  layer1, layer2, 반복횟수 순서
    (1, 512, 1, 0),
    (3, 1024, 1, 1),
    "M",
    [(1, 512, 1, 0), (3, 1024, 1, 1), 2],    # 5번째
    # (3, 1024, 1, 1),
    # (3, 1024, 2, 1),
    # (3, 1024, 1, 1),    # 6번째
    # (3, 1024, 1, 1),
]

class CNNBlock(nn.Module):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(CNNBlock, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        # 뒤에 batch norm이 나오기 때문에 bais=False로
        # 만약 bias=True이면 f(x) = w*x + b에서 b를 살려두는 것
        self.batchnorm = nn.BatchNorm2d(out_channels)
        self.leakyrelu = nn.LeakyReLU(0.1)

    def forward(self, x):
        return self.leakyrelu(self.batchnorm(self.conv(x)))

class Yolov1(nn.Module):
    def __init__(self, in_channels=3, **kwargs):
        super(Yolov1, self).__init__()
        self.architecture = architecture_config
        self.in_channels = in_channels
        self.darknet = self._create_conv_layers(self.architecture)
        self.fcs = self._create_fcs(**kwargs)
    
    def forward(self, x):
        x = self.darknet(x)
        return self.fcs(torch.flatten(x, start_dim=1))

    def _create_conv_layers(self, architecture):
        layers = []
        in_channels = self.in_channels

        for x in architecture:
            if type(x) == tuple:
                layers = layers + [
                    CNNBlock(
                        in_channels, x[1], kernel_size=x[0], stride=x[2], padding=x[3],
                    )
                ]
                in_channels = x[1]
            
            elif type(x) == str:
                # layers = layers + [nn.MaxPool2d(kernel_size=2, stride=2)]
                layers = layers + [nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))]

            elif type(x) == list:
                conv1 = x[0]    # tuple
                conv2 = x[1]    # tuple
                num_repeats = x[2]  # int

                for _ in range(num_repeats):
                    layers = layers + [
                        CNNBlock(
                            in_channels,
                            conv1[1],
                            kernel_size=conv1[0],
                            stride=conv1[2],
                            padding=conv1[3],
                        )
                    ]

                    layers = layers +[
                        CNNBlock(
                            conv1[1],
                            conv2[1],
                            kernel_size=conv2[0],
                            stride=conv2[2],
                            padding=conv2[3],
                        )
                    ]

                    in_channels = conv2[1]
        return nn.Sequential(*layers)

    def _create_fcs(self, split_size, num_boxes, num_classes):
        S, B, C = split_size, num_boxes, num_classes
        return nn.Sequential(
            nn.Flatten(),
            nn.Linear(1024*S*S, 4096),
            nn.Dropout(0.5),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S*S*(C+B*5)),

        )

# test용도
# def test(S=7, B=2, C=20):
#     model = Yolov1(split_size=S, num_boxes=B, num_classes=C)
#     # x = torch.randn((2,3,448,448))
#     x = torch.randn((2,3,224,224))
#     print(model(x).shape)
# test()