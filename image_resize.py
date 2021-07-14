"""
dataset.py에 합침!!
"""

from numpy.lib.type_check import imag
import pandas as pd
import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET

img_dir = './data/temp/train/JPEGImages/000005.jpg'

img = cv2.imread(img_dir, cv2.IMREAD_COLOR)

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

img_resize = fix_size(img)
# print(img_resize.shape)
cv2.imshow('temp', img_resize)
cv2.waitKey(0)
cv2.destroyAllWindows()
