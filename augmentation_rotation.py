import numpy as np
import cv2
from random import *
import xml.etree.ElementTree as ET
import glob

# parameter
images = glob.glob('./data/temp/train/JPEGImages/*.jpg')
XML_dir = glob.glob('./data/temp/train/Annotations/*.xml')
rotate_aug_num = 3

def transform(img): # transform(불러온 이미지 파일)
    h, w = img.shape[:2]
    theta = randint(1,360)
    M = cv2.getRotationMatrix2D((w/2, h/2), theta, 1)
    img = cv2.warpAffine(img, M, (w,h))
    return img, M, theta

for num in range(rotate_aug_num):
    i = 0
    for fname in images:
        img = cv2.imread(fname, cv2.IMREAD_COLOR)
        xmlname = XML_dir[i]
        i = i+1
        tree = ET.parse(xmlname)
        root = tree.getroot()
        rotate_img, M, theta = transform(img)

        for object in root.findall('object'):
            class_name = object.find('name').text
            for bndbox in object.findall('bndbox'):
                x1 = int(bndbox.find('xmin').text)
                y1 = int(bndbox.find('ymin').text)
                point_1_before = np.array([x1,y1,1]).reshape(3,1)
                x2 = int(bndbox.find('xmax').text)
                y2 = int(bndbox.find('ymax').text)
                point_2_before = np.array([x2,y2,1]).reshape(3,1)
                point_12 = np.array([x1,y2,1]).reshape(3,1)
                point_21 = np.array([x2,y1,1]).reshape(3,1)
                    
                point_1 = np.matmul(M, point_1_before)
                point_2 = np.matmul(M, point_2_before)
                point_12 = np.matmul(M, point_12)
                point_21 = np.matmul(M, point_21)
                
                x1 = int(np.min(np.array([point_1[0], point_2[0], point_12[0], point_21[0]])))
                x2 = int(np.max(np.array([point_1[0], point_2[0], point_12[0], point_21[0]])))

                y1 = int(np.min(np.array([point_1[1], point_2[1], point_12[1], point_21[1]])))
                y2 = int(np.max(np.array([point_1[1], point_2[1], point_12[1], point_21[1]])))
                
                draw = cv2.rectangle(rotate_img, (x1, y1), (x2, y2), (0, 0, 255), 2)
                font = cv2.FONT_HERSHEY_SIMPLEX
                cv2.putText(rotate_img, class_name, (int((x1+x2)/2), y1), font, 0.5, (255,255,255), 1)
        cv2.imshow('img with box', rotate_img)
        cv2.waitKey(0)

cv2.destroyAllWindows()