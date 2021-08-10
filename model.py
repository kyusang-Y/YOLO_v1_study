import torch.nn as nn
import torch
from torchvision.models.resnet import resnet50

model_temp = resnet50(pretrained=True)
for param in model_temp.parameters():
    param.requires_grad = False

pre_trained = nn.Sequential(*(list(model_temp.children())[0:8]))

class Yolov1(nn.Module):
    def __init__(self, pre_trained_model):
        super(Yolov1, self).__init__()
        self.resnet = pre_trained_model
        self.conv1 = nn.Conv2d(2048, 1024, 3, 1, 1)
        self.conv2 = nn.Conv2d(1024, 1024, 3, 2, 1)
        self.conv3 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.conv4 = nn.Conv2d(1024, 1024, 3, 1, 1)
        self.batchnorm = nn.BatchNorm2d(1024)
        self.leakyrelu = nn.LeakyReLU(0.1)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(1024*7*7, 4096)
        self.dropout = nn.Dropout(0.5)            
        self.linear2 = nn.Linear(4096, 7*7*(20+2*5))

    def forward(self, x):
        x = self.resnet(x)
        x = self.leakyrelu(self.batchnorm(self.conv1(x)))
        x = self.leakyrelu(self.batchnorm(self.conv2(x)))
        x = self.leakyrelu(self.batchnorm(self.conv3(x)))
        x = self.leakyrelu(self.batchnorm(self.conv4(x)))
        x = self.flatten(x)
        x = self.leakyrelu(self.dropout(self.linear1(x)))   
        x = self.linear2(x)
    
        return x

# # test용도
# model = FinetuneResnet(pre_trained)
# x = torch.randn((2,3,448,448))
# print(model(x).shape)
