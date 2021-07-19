import torch
import torchvision.transforms as transforms
import torch.optim as optim
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import VOCDataset
import os
from loss import YoloLoss
from tqdm import tqdm

# Hyperparameters etc. 
LEARNING_RATE = 2e-5
# DEVICE = "cuda" if torch.cuda.is_available else "cpu"
DEVICE = "cpu"
BATCH_SIZE = 1 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 10
NUM_WORKERS = 0
train_dir = './data/temp/train/JPEGImages'
train_files = os.listdir(train_dir)
xml_dir = './data/temp/train/Annotations'
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

def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
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

    filename_pth = 'save_model.pth'
    torch.save(model, filename_pth) # 학습시킨 model 저장!!

if __name__ =="__main__":
    main()