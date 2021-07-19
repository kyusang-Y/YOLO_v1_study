import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms
import os
import cv2
from model_test import Yolov1
test_dir = './data/temp/test'
test_files = os.listdir(test_dir)

class VOCDataset_test(torch.utils.data.Dataset):
    def __init__(self, file_list, file_dir, mode ='test', 
    S=7, B=2, C=20, transform = None):
        self.file_list = file_list
        self.file_dir = file_dir
        self.mode = mode
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):  # 왜 반드시 선언해야하는지  <- notion 참고!
        return len(self.file_list)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.file_dir, self.file_list[index]), cv2.IMREAD_COLOR) 
        
        if self.transform:
            img = self.transform(img)
        
        return img.astype('float32'), self.file_list[index]


test_transform = transforms.Compose([
    transforms.Resize((448,448)),
    transforms.ToTensor()
])

testset = VOCDataset_test(test_files, test_dir, mode='test', transform = test_transform)
testloader = DataLoader(testset, batch_size = 1, shuffle=False, num_workers=0)

filename_pth = 'save_model.pth'
model = torch.load(filename_pth)
model.eval()
# print(model)

DEVICE = 'cpu'

def test_fn(test_loader, model):
    for (x, y) in enumerate(test_loader):
        x, y = x.to(DEVICE), y.to(DEVICE)
        output = y
        print(output)
        
def main():
    model = Yolov1(split_size=7, num_boxes=2, num_classes=20).to(DEVICE)
    
    test_dataset = VOCDataset_test(
        test_files, test_dir, transform=transforms.ToTensor
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=1, shuffle=False, num_workers=0)
    
    test_fn(test_loader, model)
    
if __name__ =="__main__":
    main()



