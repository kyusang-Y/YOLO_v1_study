import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
import os
from loss import YoloLoss
from tqdm import tqdm
import torch.nn as nn
import torchvision
from torchvision.models.resnet import resnet50

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
# DEVICE = "cuda" if torch.cuda.is_available else "cpu"
DEVICE = "cpu"
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 10   # 1000
NUM_WORKERS = 0
# train_dir = './data/temp/train_sample/JPEGImages'
train_dir = './data/VOCdevkit/train/VOCdevkit/VOC2007/JPEGImages'
train_files = os.listdir(train_dir)
# xml_dir = './data/temp/train_sample/Annotations'
xml_dir = './data/VOCdevkit/train/VOCdevkit/VOC2007/Annotations'
xml_files = os.listdir(xml_dir)

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        out = model(x)
        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

class Identity(nn.Module):
    def __init__(self):
        super(Identity, self).__init__()

    def forward(self, x):
        return x

class net(nn.Module):
    def __init__(self):
        super(net, self).__init__()
        

model_temp = torchvision.models.resnet50(pretrained=True)
for param in model_temp.parameters():
    param.requires_grad = False

pre_trained = nn.Sequential(*(list(model_temp.children())[0:8]))
# print(features)

class FinetuneResnet(nn.Module):
    def __init__(self, pre_trained_model):
        super(FinetuneResnet, self).__init__()
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



def main():
    model = FinetuneResnet(pre_trained).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    loss_fn = YoloLoss()

    train_dataset = VOCDataset(
        train_files, train_dir, xml_files, xml_dir, transform=transforms.ToTensor
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
        
    for epoch in range(EPOCHS):    
        train_fn(train_loader, model, optimizer, loss_fn)

    filename_pth = 'save_model_many.pth'
    torch.save(model, filename_pth) # 학습시킨 model 저장!!

if __name__ =="__main__":
    main()

