import torch
from torch.utils import data
import torchvision.transforms as transforms
import torch.optim as optim
import os
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
from loss import loss_fn

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
GPU_NUM = 2 # 원하는 GPU 번호 입력
DEVICE = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(DEVICE) # change allocation of current GPU

# DEVICE = "cpu"
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0.0005
EPOCHS = 500   # 135
NUM_WORKERS = 4
# train_dir = './data/temp/train_sample/JPEGImages'   # 1장
# train_dir = './data/temp/train/JPEGImages'    # 14장
train_dir = './data/VOCdevkit/train/VOCdevkit/VOC2007/JPEGImages'   # 여러장
train_files = os.listdir(train_dir)
# xml_dir = './data/temp/train_sample/Annotations'
# xml_dir = './data/temp/train/Annotations'
xml_dir = './data/VOCdevkit/train/VOCdevkit/VOC2007/Annotations'
xml_files = os.listdir(xml_dir)

def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to(DEVICE), y.to(DEVICE)
        # print(y)
        out = model(x)
        # out = torch.clamp(out, min=0)
        loss = loss_fn(out, y, BATCH_SIZE)  # prediction과 target
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # update progress bar
        loop.set_postfix(loss=loss.item())

    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)}")

from torchvision.models.resnet import resnet50
import torch.nn as nn
model_temp = resnet50(pretrained=True)
for param in model_temp.parameters():
    param.requires_grad = False

pre_trained = nn.Sequential(*(list(model_temp.children())[0:8]))

def main():
    model = Yolov1(pre_trained).to(DEVICE)
    optimizer = optim.Adam(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    train_dataset = VOCDataset(
        train_files, train_dir, xml_files, xml_dir, transform=transforms.ToTensor
    )

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    for epoch in range(EPOCHS):    
        train_fn(train_loader, model, optimizer, loss_fn)

    filename_pth = 'save_weight.pth'
    torch.save(model, filename_pth) # 학습시킨 model 저장!!

if __name__ =="__main__":
    main()

