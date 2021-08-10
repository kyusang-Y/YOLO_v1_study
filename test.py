import torch
from torch.utils.data import DataLoader
import os
import cv2
import numpy as np
import torch.nn as nn
from train import NUM_WORKERS, BATCH_SIZE
class_dict = {"aeroplane":1, "bicycle":2, "bird":3, "boat":4, "bottle":5, 
"bus":6, "car":7, "cat":8, "chair":9, "cow":10, 
"diningtable":11, "dog":12, "horse":13, "motorbike":14, "person":15,
"pottedplant":16, "sheep":17, "sofa":18, "train":19, "tvmonitor":20}

list_of_key = list(class_dict.keys())
list_of_value = list(class_dict.values())

# test_dir = './data/temp/train/JPEGImages'
# test_dir = './data/temp/train_sample/JPEGImages'   # 1장
test_dir = './data/temp/train/JPEGImages'    # 14장
# test_dir = './data/VOCdevkit/test/VOCdevkit/VOC2007/JPEGImages'
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
testloader = DataLoader(testset, batch_size = BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)

filename_pth = 'save_weight.pth'
model = torch.load(filename_pth)
model.eval()
threshold = 0.5
image_index = 0

GPU_NUM = 2 # 원하는 GPU 번호 입력
DEVICE = torch.device(f'cuda:{GPU_NUM}' if torch.cuda.is_available() else 'cpu')
torch.cuda.set_device(DEVICE) # change allocation of current GPU

for img_whole, index in testloader:
    img_whole = img_whole.to(DEVICE)
    # print(index)
    for batchindex in range(img_whole.shape[0]):
        img = img_whole[batchindex,...].unsqueeze(0)
        output = model(img)
        output = torch.reshape(output, (7,7,30))
        # print("output tensor size : ", output.shape)
        
        img = img.cpu().detach().squeeze()
        img = np.transpose(img, (1,2,0)).numpy()    # opencv에 맞게 수정, 근데 0~1사이의 값인데도 잘 출력이니 냅두기
        img = img * 255
        img = img.astype(np.uint8)
        img = np.ascontiguousarray(img, dtype=np.uint8)
        # print(img.shape)
        output = output.cpu().detach().numpy()
        confidence1 = output[...,20]
        confidence2 = output[...,25]
        # print("confidence1 : ", confidence1)
        # print("confidence2 : ", confidence2)
        where1 = np.transpose(np.nonzero(confidence1 > threshold))
        # print(where1)
        where2 = np.transpose(np.nonzero(confidence2 > threshold))
        # print(output.shape)
        for i in range(len(where1)):
            x_grid = where1[i,0]
            # print('x grid', x_grid)
            y_grid = where1[i,1]

            class_index = np.argmax(output[x_grid,y_grid,:20])+1 # index를 맞추기 위하여 +1
            # print(class_index)
            grid_size = pixel_size / 7
            # print(output[x_grid,y_grid,20:25])
            x = output[x_grid,y_grid,21] * grid_size + x_grid*grid_size
            y = output[x_grid,y_grid,22] * grid_size + y_grid*grid_size
            w = output[x_grid,y_grid,23] * grid_size
            h = output[x_grid,y_grid,24] * grid_size
            # x = pixel_size/7 * x + output
            # print(x, y, w, h)
            draw = cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)),
            (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = list_of_value.index(class_index)
            # print("1번", list_of_key[position])
            cv2.putText(img, list_of_key[position], (int(x-w/2), int(y-h/2)), font, 0.5, (255,255,255), 1)
        
        for i in range(len(where2)):
            x_grid = where2[i,0]
            y_grid = where2[i,1]

            class_index = np.argmax(output[x_grid,y_grid,:20])+1 # index를 맞추기 위하여 +1
            # print(class_index)
            grid_size = pixel_size / 7
            # print(output[x_grid,y_grid,25:30])
            x = output[x_grid,y_grid,26] * grid_size + x_grid*grid_size
            y = output[x_grid,y_grid,27] * grid_size + y_grid*grid_size
            w = output[x_grid,y_grid,28] * pixel_size
            h = output[x_grid,y_grid,29] * pixel_size
            # x = pixel_size/7 * x + output
            # print(x, y, w, h)
            draw = cv2.rectangle(img, (int(x-w/2), int(y-h/2)), (int(x+w/2), int(y+h/2)),
            (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = list_of_value.index(class_index)
            # print("2번", list_of_key[position])
            cv2.putText(img, list_of_key[position], (int(x-w/2), int(y-h/2)), font, 0.5, (255,255,255), 1)

        cv2.imwrite(f'image{image_index}.jpg', img)   # 서버에서 저장하는 용도로 사용!
        # cv2.imwrite(f'{index[batchindex]}', img)
        image_index = image_index+1
print("finished!") 

