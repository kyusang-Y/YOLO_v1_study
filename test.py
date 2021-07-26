import torch
from torch._C import dtype
from torch.nn.modules.activation import Threshold
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from torchvision import transforms, models
import os
import cv2
import numpy as np
from PIL import Image
from torch.optim import lr_scheduler
import torch.nn as nn
import torch.optim as optim
from train_1 import FinetuneResnet

class_dict = {"aeroplane":1, "bicycle":2, "bird":3, "boat":4, "bottle":5, 
"bus":6, "car":7, "cat":8, "chair":9, "cow":10, 
"diningtable":11, "dog":12, "horse":13, "motorbike":14, "person":15,
"pottedplant":16, "sheep":17, "sofa":18, "train":19, "tvmonitor":20}

list_of_key = list(class_dict.keys())
list_of_value = list(class_dict.values())

test_dir = './data/temp/test'
test_files = os.listdir(test_dir)
pixel_size = 448
class VOCDataset_test(torch.utils.data.Dataset):
    def __init__(self, file_list, file_dir, mode ='test', 
    S=7, B=2, C=20, transform = None):
        self.file_list = file_list
        self.file_dir = file_dir
        self.mode = mode
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):  # 왜 반드시 선언해야하는지  <- notion 참고!
        # print(len(self.file_list))
        return len(self.file_list)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.file_dir, self.file_list[index]), cv2.IMREAD_COLOR) 
        img_height, img_width = img.shape[:2]

        if img_height > img_width:
            des_width = int(img_width*pixel_size/img_height)
            des_height = pixel_size
        else:
            des_width = pixel_size
            des_height = int(img_height*pixel_size/img_width)
        
        x_offset = (pixel_size-des_width) // 2
        y_offset = (pixel_size-des_height) // 2

        img_return = np.full((pixel_size,pixel_size,3), 127, dtype=np.int32)  # 224 * 224 size로 변환!
        img_resize = cv2.resize(img, (des_width, des_height), interpolation = cv2.INTER_AREA)
        img_return[y_offset:y_offset+des_height, x_offset:x_offset+des_width] = img_resize

        # img_return = cv2.resize(img, (pixel_size,pixel_size), interpolation=cv2.INTER_NEAREST)
        img_return = np.transpose(img_return, (2,0,1)) # torch는 channel width height이고 opencv는 width height channel
        img_return = torch.from_numpy(img_return).float().div(255.0)    #flaot형태로
        return img_return, self.file_list[index]

testset = VOCDataset_test(test_files, test_dir, mode='test')
testloader = DataLoader(testset, batch_size = 1, shuffle=False, num_workers=0)

filename_pth = 'save_model_sample.pth'
model = torch.load(filename_pth)
model.eval()
# print(model)
threshold = 0.2
image_index = 0
DEVICE = 'cpu'
for img, index in testloader:
    img = img.to(DEVICE)
    output = model(img)
    torch.set_printoptions(threshold=10000)   # input torch 확인용
    # print(output.shape)
    # output = output.squeeze()
    output = torch.reshape(output, (7,7,30))
    # print("output tensor size : ", output.shape)

    img = img.squeeze() # batch size인 차원 삭제
    img = np.transpose(img, (1,2,0)).numpy()    # opencv에 맞게 수정, 근데 0~1사이의 값인데도 잘 출력이니 냅두기
    img = img * 255
    img = img.astype(np.uint8)
    img = np.ascontiguousarray(img, dtype=np.uint8)

    output = output.detach().numpy()
    confidence1 = output[...,20]
    # print(confidence1)
    confidence2 = output[...,25]
    where1 = np.transpose(np.nonzero(confidence1 > threshold))
    # print(where1)

    where2 = np.transpose(np.nonzero(confidence2 > threshold))
    # print(where2)


    for i in range(len(where1)):
        x_grid = where1[i,0]
        y_grid = where1[i,1]

        class_index = np.argmax(output[x_grid,y_grid,:20])+1 # index를 맞추기 위하여 +1
        # print(class_index)
        grid_size = pixel_size / 7
        print(output[x_grid,y_grid,21:25])
        x = output[x_grid,y_grid,21] * grid_size + x_grid*grid_size
        y = output[x_grid,y_grid,22] * grid_size + y_grid*grid_size
        w = output[x_grid,y_grid,23] * pixel_size
        h = output[x_grid,y_grid,24] * pixel_size
        # x = pixel_size/7 * x + output
        # print(x, y, w, h)
        draw = cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)),
        (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        position = list_of_value.index(class_index)
        cv2.putText(img, list_of_key[position], (int(x-w/2), int(y-h/2)), font, 0.5, (255,255,255), 1)
    cv2.imshow('img', img)
    # cv2.imwrite(f'image{image_index}.jpg', img)   # 서버에서 저장하는 용도로 사용!
    image_index = image_index+1
    cv2.waitKey(0)

# confidence, indice = torch.max(output[...,:20], dim=2)
# confidence = confidence.detach().numpy()
# indice = indice.detach().numpy() + 1 # 0부터 시작하는 index를 맞춘 것


  
cv2.destroyAllWindows()


