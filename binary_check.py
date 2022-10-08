import cv2
import numpy as np
from matplotlib import pyplot as plt
import os

def dir_array(dir_path):
    itemlist = os.listdir(dir_path)
    dir_list = []
    # 獲取目錄檔案列表
    for item in itemlist:
        # 連線成完整路徑
        item_path = os.path.join(dir_path, item)
        dir_list.append(item_path) 
    
    return dir_list

binary_path = "0502\\comb_output.jpg"
img_path = "0502\\image_0096_cloth.jpg"

'''binaryImg = dir_array(binay_path)
print(binaryImg)
Img = dir_array(img_path)
print(Img)
'''
binary_img = cv2.imread(binary_path)
img = cv2.imread(img_path)

kernel = np.ones((1,1), np.uint8)
binary_img = cv2.dilate(binary_img, kernel, iterations = 1)

#cv2.imshow('binary_img', binary_img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

output = img.copy()
height, weight = img.shape[:2]
for i in range(height):
    for j in range(weight):
        #print(binary_img[i][j])
        pix = 1
        if binary_img[i][j][0] <= 127 or binary_img[i][j][1] <= 127 or binary_img[i][j][2] <=127 :
            pix = 0
        #print(np.mean(grabcut_output[i][j]))
        #pix = np.mean(binary_img[i][j], dtype = int)
        #if pix != 0 and pix != 255:
        #    print(pix)
        if pix == 0:
            output[i][j] = (0, 0, 0)
#cv2.imshow('Input', img)
cv2.imwrite("0502\\comb_check.jpg", output)
cv2.imshow('Result', output)
cv2.waitKey(0)