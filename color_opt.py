
# color_opt.py

from re import L
import cv2
from cv2 import bitwise_and
from cv2 import bitwise_not
from cv2 import cvtColor
import numpy as np
import dlib
import time
import os
import pathlib
from PIL import Image

import matplotlib.pyplot as plt
from Make_landmarks import MakeLandmarks
from BinaryMaskClass import mask_binary
from MakePointClass import Mask_points

MP = Mask_points()

def getavgstd(image, height, width):
    avg = []
    std = []
    image_avg_l = 0
    image_avg_a = 0
    image_avg_b = 0
    image_std_l = []
    image_std_a = []
    image_std_b = []
    num = 0

    for i in range(height) :
        for j in range(width) :
            if image[i][j][0] == 0 and image[i][j][1] == 128 and image[i][j][2] == 128 :
                continue
            else :
                image_avg_l += image[i,j,0]
                image_std_l.append(image[i,j,0])
                image_avg_a += image[i,j,1]
                image_std_a.append(image[i,j,1])
                image_avg_b += image[i,j,2]
                image_std_b.append(image[i,j,2])
                num += 1
    image_avg_l = image_avg_l/num
    image_avg_a = image_avg_a/num
    image_avg_b = image_avg_b/num
    std_l = np.std(image_std_l, axis = 0, ddof=1)
    std_a = np.std(image_std_a, axis = 0, ddof=1)
    std_b = np.std(image_std_b, axis = 0, ddof=1)
    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(std_l)
    std.append(std_a)
    std.append(std_b)
    print(std_l, std_a, std_b)
    return (avg,std)

def dir_array(dir_path):
    itemlist = os.listdir(dir_path)
    dir_list = []
    # 獲取目錄檔案列表
    for item in itemlist:
        # 連線成完整路徑
        item_path = os.path.join(dir_path, item)
        dir_list.append(item_path) 
    
    return dir_list

output_file = "0605person3"
input_file = "0605person3\color_inputImg"
MaskBinary_file = "0605person3\color_inputBinary"

input_path = "0605person3\\128\\noTrans.jpg"
MaskBinary_path = "0605person3\\128\\mask.jpg"
Rem_path = "0605person3\\128\\diff_rem.jpg"
#LandmarksBinary_path = "0605person3\\128\\masked_landmarks.jpg"
Unmasked_path = "0605person3\\128\\unmasked_img.jpg"
Masked_path = "0605person3\\128\\masked_img.jpg"


def cr_otsu(img):
    """YCrCb顏色空間的Cr分量+Otsu閾值分割"""
    # img = cv2.imread(image, cv2.IMREAD_COLOR)
    ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCR_CB)

    (y, cr, cb) = cv2.split(ycrcb)
    cr1 = cv2.GaussianBlur(cr, (5, 5), 0)
    _, skin = cv2.threshold(cr1,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    '''cv2.namedWindow("image raw", cv2.WINDOW_NORMAL)
    cv2.imshow("image raw", img)
    cv2.namedWindow("image CR", cv2.WINDOW_NORMAL)
    cv2.imshow("image CR", cr1)
    cv2.namedWindow("Skin Cr+OTSU", cv2.WINDOW_NORMAL)
    cv2.imshow("Skin Cr+OTSU", skin)'''

    dst = cv2.bitwise_and(img, img, mask=skin)
    # cv2.namedWindow("seperate", cv2.WINDOW_NORMAL)
    cv2.imshow("seperate", dst)
    cv2.waitKey(0)
    return dst

def crcb_range_sceening(img):
    # img = cv2.imread(image,cv2.IMREAD_COLOR)
    ycrcb=cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    (y,cr,cb)= cv2.split(ycrcb)

    skin = np.zeros(cr.shape,dtype= np.uint8)
    (x,y)= cr.shape
    for i in range(0,x):
        for j in range(0,y):
            if (cr[i][j]>140)and(cr[i][j])<175 and (cr[i][j]>100) and (cb[i][j])<120:
                skin[i][j]= 255
            else:
                skin[i][j] = 0
    dst = cv2.bitwise_and(img,img,mask=skin)
    #cv2.namedWindow("cutout",cv2.WINDOW_NORMAL)
    #cv2.imshow("cutout",dst)
    return dst

basename = os.path.basename(input_path)
outputname = os.path.splitext(basename)[0]
print(outputname)
img = cv2.imread(input_path)
Mask = cv2.imread(MaskBinary_path)
Rem_Mask = cv2.imread(Rem_path)
#LandmarksMask = cv2.imread(LandmarksBinary_path)
height, width, channels = img.shape

for j in range(height) :
    for k in range(width) :
        if Mask[j][k][0] > 127 and Mask[j][k][1] > 127 and Mask[j][k][2] > 127 :
            Mask[j][k] = 255
        else :
            Mask[j][k] = 0

for j in range(height) :
    for k in range(width) :
        if Rem_Mask[j][k][0] > 127 and Rem_Mask[j][k][1] > 127 and Rem_Mask[j][k][2] > 127 :
            Rem_Mask[j][k] = 255
        else :
            Rem_Mask[j][k] = 0

'''for j in range(height) :
    for k in range(width) :
        if LandmarksMask[j][k][0] > 127 and LandmarksMask[j][k][1] > 127 and LandmarksMask[j][k][2] > 127 :
            LandmarksMask[j][k] = 255
        else :
            LandmarksMask[j][k] = 0'''


            


#LandmarksMask = cv2.cvtColor(LandmarksMask, cv2.COLOR_BGR2GRAY)
Mask = cv2.cvtColor(Mask, cv2.COLOR_BGR2GRAY)
Rem_Mask = cv2.cvtColor(Rem_Mask, cv2.COLOR_BGR2GRAY)
cv2.imshow("Rem_Mask", Rem_Mask)
cv2.waitKey(0)
'''print(Mask)
for j in range(height) :
    for k in range(width) :
        if Mask[j][k] <= 1:
            Mask[j][k] = 0'''

org_img = bitwise_and(img, img, mask = Mask)
# opp_Mask = cv2.bitwise_not(Mask)
# ref_img = bitwise_and(img, img, mask = LandmarksMask)
# ref_img = cv2.bitwise_and(ref_img, ref_img, mask = opp_Mask)
ref_img = bitwise_and(img, img, mask = Rem_Mask)
org_skin = cr_otsu(org_img)
ref_skin = cr_otsu(ref_img)
cv2.imshow("org_skin", org_skin)
cv2.waitKey(0)
cv2.imshow("ref_skin", ref_skin)
cv2.waitKey(0)
cv2.destroyAllWindows()
'''cv2.imshow("org_img", org_img)
cv2.waitKey(0)
cv2.imshow("ref_img", ref_img)
cv2.waitKey(0)
cv2.destroyAllWindows()'''
img_LAB = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
org_img_LAB = cv2.cvtColor(org_skin,cv2.COLOR_BGR2LAB)
ref_img_LAB = cv2.cvtColor(ref_skin,cv2.COLOR_BGR2LAB)
'''plt.imshow(ref_img_LAB)
plt.show()'''
print(org_img_LAB)

# source_avg,source_std = Sgetavgstd(ref_img_LAB, y, x, h, w) # 目標(無口罩(去對應口罩部分)圖片)
ref_avg, ref_std = getavgstd(ref_img_LAB, height, width)
org_avg, org_std = getavgstd(org_img_LAB, height, width)

img_skin = img.copy()
img_skin = cv2.cvtColor(img_skin, cv2.COLOR_BGR2LAB)
for i in range(height) :
    for j in range(width) :
        if Mask[i][j] == 255 :
            img_skin[i][j] = ref_avg
img_skin = cv2.cvtColor(img_skin, cv2.COLOR_LAB2BGR)
cv2.imshow("img_skin", img_skin)
cv2.waitKey()


#測試
chk = np.zeros_like(img)
for i in range(height) :
    for j in range(width) :
        chk[i][j] = ref_avg
chk2 = np.zeros_like(img)
for i in range(height) :
    for j in range(width) :
        chk2[i][j] = org_avg
chk  = cv2.cvtColor(chk,cv2.COLOR_LAB2BGR)
chk2  = cv2.cvtColor(chk2,cv2.COLOR_LAB2BGR)
cv2.imshow("check", chk)
cv2.waitKey(0)
cv2.imshow("check", chk2)
cv2.waitKey(0)
cv2.destroyAllWindows()

output_img = img_LAB.copy()
LAB = img_LAB.copy()
for i in range(height) :
    for j in range(width) :
        con = 1
        for z in range(3) :
            if img[i][j][z] != org_img[i][j][z] :
                con = 0
        if Mask[i][j] == 0 :
            con = 0
        if con == 1 :
            for k in range(0, 3):
                change = img_LAB[i][j][k]
                change = (change-org_avg[k])*(ref_std[k]/org_std[k]) + ref_avg[k]
                change = 0 if change<0 else change
                change = 255 if change>255 else change
                #output_img[i,j,k] = LAB[i,j,k]*0.2 + change*0.8
                output_img[i,j,k] = LAB[i,j,k]*0.5 + change*0.5
output_img = cv2.cvtColor(output_img, cv2.COLOR_LAB2BGR)


x1, y1, x2, y2 = 128, 128, 0, 0
for i in range(height) :
    for j in range(width) :
        if Mask[i][j] != 0 :
            x1 = min(i, x1)
            y1 = min(j, y1)
            x2 = max(i, x2)
            y2 = max(j, y2)

print(x1, y1, x2, y2)

img_copy = img.copy()
img_copy = bitwise_and(img, img, mask = Mask)

cv2.imshow("img_copy", img_copy)
cv2.waitKey(0)


xx = (x1 + x2) / 2
yy = (y1 + y2) / 2
center_face = (int(yy), int(xx))

cv2.imshow("output_img", output_img)
cv2.waitKey(0)

seamlessclone = cv2.seamlessClone(img_skin, output_img, Mask, center_face, cv2.MIXED_CLONE)
cv2.imshow("seamlessclone", seamlessclone)
cv2.waitKey(0)
cv2.waitKey(0)
cv2.destroyAllWindows()
# cv2.imwrite("0602_test_color\seamlessclone.jpg", seamlessclone)
'''
        if img2_new_face[i, j, 0] == 0 and img2_new_face[y + i, x + j, 1] == 0 and img2_new_face[y + i, x + j, 2] == 0:
            continue
        for k in range(0,3):
            t = img2_new_face_LAB[y+i,x+j,k]
            t = (t-source_avg[k])*(target_std[k]/source_std[k]) + target_avg[k]
            t = 0 if t<0 else t
            t = 255 if t>255 else t
            LAB[y+i,x+j,k] = t'''

cv2.imwrite(output_file + "\\" + "output_img.jpg" , output_img)
cv2.imwrite(output_file + "\\" + "seamlessclone.jpg", seamlessclone)