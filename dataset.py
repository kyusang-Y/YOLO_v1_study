from random import shuffle
from numpy.matrixlib.defmatrix import matrix
import pandas as pd
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import torch
import os
from torchvision import transforms

class_dict = {"aeroplane":1, "bicycle":2, "bird":3, "boat":4, "bottle":5, 
"bus":6, "car":7, "cat":8, "chair":9, "cow":10, 
"diningtable":11, "dog":12, "horse":13, "motorbike":14, "person":15,
"pottedplant":16, "sheep":17, "sofa":18, "train":19, "tvmonitor":20}

train_dir = './data/temp/train/JPEGImages'
train_files = os.listdir(train_dir)
xml_dir = './data/temp/train/Annotations'
xml_files = os.listdir(xml_dir)
# print(xml_files)

class VOCDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, file_dir, xml_list, xmlfile_dir, mode ='train', 
    S=7, B=2, C=20, transform = None):
        self.file_list = file_list
        self.file_dir = file_dir
        self.xml_list = xml_list
        self.xmlfile_dir = xmlfile_dir
        self.mode = mode
        self.transform = transform
        self.S = S
        self.B = B
        self.C = C

    def __len__(self):
        # print("file list 길이 : ", len(self.file_list))
        return len(self.file_list)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.file_dir, self.file_list[index]), cv2.IMREAD_COLOR) 
        img_height, img_width = img.shape[:2]
    
        if img_height > img_width:
            des_width = int(img_width*224/img_height)
            des_height = 224
        else:
            des_width = 224
            des_height = int(img_height*224/img_width)
        
        x_offset = (224-des_width) // 2
        y_offset = (224-des_height) // 2

        img_return = np.full((224,224,3), 127, dtype=np.uint8)  # 244 * 244 size로 변환!
        img_resize = cv2.resize(img, (des_width, des_height), interpolation = cv2.INTER_AREA)
        img_return[y_offset:y_offset+des_height, x_offset:x_offset+des_width] = img_resize

        label_matrix = torch.zeros((self.S, self.S, self.C + 5 * self.B))
        tree = ET.parse(os.path.join(self.xmlfile_dir, self.xml_list[index]))
        root = tree.getroot()
        label_matrix = []
        for object in root.findall('object'):
            class_name = object.find('name').text
            class_number = class_dict[class_name]
            # print(type(class_number))
            # print("class number = ", class_number)
            for bndbox in object.findall('bndbox'):
                x1 = int(bndbox.find('xmin').text)
                y1 = int(bndbox.find('ymin').text)
                x2 = int(bndbox.find('xmax').text)
                y2 = int(bndbox.find('ymax').text)
                x = ((x1+x2)/2)/img_width   # 0~1사이의 값으로 normalize
                y = ((y1+y2)/2)/img_height
                w = (x2-x1)/img_width
                h = (y2-y1)/img_height

                # print("x1 y1 x2 y2", x1, y1, x2, y2)
                
                # 중심점 좌표를 224*224 size에 맞게 변환!
                if img_height > img_width:
                    # print('h, y는 그대로')
                    x = (x_offset + (224 - x_offset*2)*x)/224
                    w = w * des_width/224 
                else:
                    # print('w ,x는 그대로')
                    y = (y_offset + (224- y_offset*2)*y)/224
                    h = h * des_height /224
                label_matrix.append([class_number, x, y, w, h]) 
                
                """
                원점이 이미지 왼쪽 위

                x1 y1이 bounding box의 왼쪽 위
                x2 y2가 bounding box의 오른쪽 아래

                (x1, y1)
                
                
                            (x2, y2)
                """
                
                ## 시각화 용도 ##
                # draw = cv2.rectangle(img_return, (int((x-w/2)*224), int((y-h/2)*224)), 
                # (int((x+w/2)*224), int((y+h/2)*224)), (0, 0, 255), 2)
                # font = cv2.FONT_HERSHEY_SIMPLEX
                # cv2.putText(img_return, class_name, (int(x*224), int((y-h/2)*224)), font, 0.5, (255,255,255), 1)  
        return img_return, label_matrix

train_temp = VOCDataset(train_files, train_dir, xml_files, xml_dir, transform=transforms.ToTensor)
data_loader = torch.utils.data.DataLoader(train_temp, batch_size=1, shuffle=False, num_workers=0)


## 시각화 용도 ##
# for img, label in data_loader:
#     # print(type(img))
#     # print(img.size())
#     img = img[0].numpy()
#     # print("img size 가로, 세로, 채널 수 : ", img.shape)
#     label_for_show = np.array(label)
#     print("label matrix : ", label_for_show)
#     cv2.imshow('img', img)
#     cv2.waitKey(0)
# cv2.destroyAllWindows()