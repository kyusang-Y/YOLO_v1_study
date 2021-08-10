from random import shuffle
import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
import torch
import os
from torchvision import transforms
from torch.utils.data import DataLoader

pixel_size = 448  
class_dict = {"aeroplane":1, "bicycle":2, "bird":3, "boat":4, "bottle":5, 
"bus":6, "car":7, "cat":8, "chair":9, "cow":10, 
"diningtable":11, "dog":12, "horse":13, "motorbike":14, "person":15,
"pottedplant":16, "sheep":17, "sofa":18, "train":19, "tvmonitor":20}

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
        return len(self.file_list)

    def __getitem__(self, index):
        img = cv2.imread(os.path.join(self.file_dir, self.file_list[index]), 
        cv2.IMREAD_COLOR) 
        img_height, img_width = img.shape[:2]
        label_matrix = torch.zeros((self.S, self.S, self.C + self.B*5))
    
        if img_height > img_width:
            des_width = int(img_width*pixel_size/img_height)
            des_height = pixel_size
        else:
            des_width = pixel_size
            des_height = int(img_height*pixel_size/img_width)
        
        x_offset = (pixel_size-des_width) // 2
        y_offset = (pixel_size-des_height) // 2

        img_return = np.full((pixel_size,pixel_size,3), 127, dtype=np.uint8)  # 224 * 224 size로 변환!
        img_resize = cv2.resize(img, (des_width, des_height), interpolation = cv2.INTER_AREA)
        img_return[y_offset:y_offset+des_height, x_offset:x_offset+des_width] = img_resize

        tree = ET.parse(os.path.join(self.xmlfile_dir, self.xml_list[index]))
        root = tree.getroot()
     
        for object in root.findall('object'):
            class_name = object.find('name').text
            class_number = class_dict[class_name]
         
            for bndbox in object.findall('bndbox'):
                x1 = int(float(bndbox.find('xmin').text))
                y1 = int(float(bndbox.find('ymin').text))
                x2 = int(float(bndbox.find('xmax').text))
                y2 = int(float(bndbox.find('ymax').text))
                x = ((x1+x2)/2)/img_width   # 0~1사이의 값으로 normalize
                y = ((y1+y2)/2)/img_height
                w = (x2-x1)/img_width * self.S
                h = (y2-y1)/img_height * self.S
                
                # 중심점 좌표를 448*448 size에 맞게 변환!
                if img_height > img_width:
                    # print('h, y는 그대로')
                    x = (x_offset + (pixel_size - x_offset*2)*x)/pixel_size
                    w = w * des_width/pixel_size 
            
                else:
                    # print('w ,x는 그대로')
                    y = (y_offset + (pixel_size- y_offset*2)*y)/pixel_size
                    h = h * des_height /pixel_size

                i = int(self.S*x)
                if i>=7:
                    i = 6

                j = int(self.S*y)
                if j>=7:
                    j = 6

                x_cell = self.S*x - i 
                # grid를 경계로 0~1 사이 값으로 정규화 but why??
                # grid별로 bounding box를 2개씩 예측하기 때문인듯
                y_cell = self.S*y - j
                if label_matrix[i, j, 20] == 0: 
                    label_matrix[i, j, class_number-1] = 1
                    label_matrix[i, j, 20] = 1
                    label_matrix[i, j, 21] = x_cell
                    label_matrix[i, j, 22] = y_cell
                    label_matrix[i, j, 23] = w
                    label_matrix[i, j, 24] = h

        img_return = np.transpose(img_return, (2,0,1))  
        # torch는 channel width height이고 opencv는 width height channel
        img_return = torch.from_numpy(img_return).float().div(255.0)    # 학습시킬려면 flaot형태로
        return img_return, label_matrix
