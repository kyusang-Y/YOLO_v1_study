import pandas as pd
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

images = glob.glob('./data/temp/train/JPEGImages/*.jpg')

i = 0
XML_dir = glob.glob('./data/temp/train/Annotations/*.xml')

def fix_size(img, fill_color=(192,192,192)):
    tmp = np.full((224,224,3), 127, dtype=np.uint8)
    row, col = img.shape[:2]
    if row > col:
        des_width = int(col*224/row)
        des_height = 224
    else:
        des_width = 224
        des_height = int(row*224/col)
        
    img_resize = cv2.resize(img, (des_width, des_height), interpolation = cv2.INTER_AREA)
    x_offset = (224-des_width) // 2
    y_offset = (224-des_height) // 2
    tmp[y_offset:y_offset+des_height, x_offset:x_offset+des_width] = img_resize
    # return img_resize
    return tmp

for fname in images:
    img = cv2.imread(fname, cv2.IMREAD_COLOR)
    # print(img.shape)
    xmlname = XML_dir[i]
    i = i+1
    tree = ET.parse(xmlname)
    root = tree.getroot()
    # print("root : ", root.tag)
    for object in root.findall('object'):
        class_name = object.find('name').text
        for bndbox in object.findall('bndbox'):
            x1 = int(bndbox.find('xmin').text)
            y1 = int(bndbox.find('ymin').text)
            x2 = int(bndbox.find('xmax').text)
            y2 = int(bndbox.find('ymax').text)
            # print("x1 y1 x2 y2", x1, y1, x2, y2)            
            """
            원점이 이미지 왼쪽 위

            x1 y1이 bounding box의 왼쪽 위
            x2 y2가 bounding box의 오른쪽 아래

            (x1, y1)
            
            
                        (x2, y2)
            """

            draw = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(img, class_name, (int((x1+x2)/2), y1), font, 0.5, (255,255,255), 1)  
    img = fix_size(img)
    print(img.shape)
    cv2.imshow('img with box', img)
    cv2.waitKey(0)
cv2.destroyAllWindows()
