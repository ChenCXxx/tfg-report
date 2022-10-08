import numpy as np
import cv2
from matplotlib import pyplot as plt
from Make_landmarks import MakeLandmarks
import dlib
from PIL import Image
import os

class Point(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def getX(self):
        return self.x

    def getY(self):
        return self.y

def getGrayDiff(img, currentPoint, tmpPoint):
        return abs((img[currentPoint.x, currentPoint.y]).astype(int) - (img[tmpPoint.x, tmpPoint.y]).astype(int))


def selectConnects(p):
    if p != 0:
        connects = [Point(-1, -1), Point(0, -1), Point(1, -1), Point(1, 0), Point(1, 1), Point(0, 1), Point(-1, 1), Point(-1, 0)]
    else:
        connects = [Point(0, -1), Point(1, 0), Point(0, 1), Point(-1, 0)]
    return connects

def regionGrow(img, seeds, thresh, p=1):
    height, weight = img.shape[:2]
    seedMark = np.zeros(img.shape)
    output = np.zeros(img.shape)
    seedList = []
    for seed in seeds:
        seedList.append(seed)
        st = seed
    label = (255, 255, 255)
    #print(seedList)
    connects = selectConnects(p)
    while (len(seedList) > 0):
        currentPoint = seedList.pop(0)
        #print(currentPoint)
        seedMark[currentPoint.x, currentPoint.y] = label
        for i in range(8):
            tmpX = currentPoint.x + connects[i].x
            tmpY = currentPoint.y + connects[i].y
            if tmpX < 0 or tmpY < 0 or tmpX >= height or tmpY >= weight:
                continue
            grayDiff = getGrayDiff(img, st, Point(tmpX, tmpY))
            
            grayDiff = grayDiff.mean(axis=0).astype("int")
            if (grayDiff <= thresh) and (seedMark[tmpX, tmpY].all() == 0):
                #print([tmpX, tmpY])
                seedMark[tmpX, tmpY] = label
                seedList.append(Point(tmpX, tmpY))
        #plt.imshow(seedMark)
        #plt.show()
    return seedMark

def show(img):
    #轉換通道
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_)

#執行Grabcut函數
def grabcut(img, mask, rect, iters=20):
    img_ = img.copy()
    bg_model = np.zeros((1, 65),np.float64)
    fg_model = np.zeros((1, 65),np.float64)
    cv2.grabCut(img.copy(), mask, rect, bg_model, fg_model, iters, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_ = img*mask2[:,:,np.newaxis]
    return img_

def make_binary_image(img):
    height, weight = img.shape[:2]
    output = np.zeros(img.shape)
    for i in range(height):
        for j in range(weight):
            if(img[i][j].all() != 0):
                output[i][j] = (255, 255, 255)
    return output


class mask_binary():
    
    
    def find_mask_binary(self, MaskImg_path, NoMaskImg_path, file_dir, filename):
        name = MaskImg_path[-14:-10]
        img = cv2.imread(MaskImg_path)
        img2 = cv2.imread(NoMaskImg_path)
        #顯示原始圖片
        #filename = NoMaskImg_path[-8:-4]
        output_path = file_dir + "\\" + filename + "_binary" + ".jpg"

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

        detector = dlib.get_frontal_face_detector()
        predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
        faces = detector(img_gray)
        faces2 = detector(img2_gray)

        x1, y1, x2, y2 = 592, 592, 0, 0
        flag = False
        if not faces :
            print("faces")
            return None
        if not faces2 :
            print("faces2")
            return None
        for i in range(len(faces)):
            flag = True
            face = faces[i]
            face2 = faces2[i]
            landmarks = predictor(img_gray, face)
            ML = MakeLandmarks(predictor)
            points = ML.make_landmarks_align(img2, img2_gray, face2, img, img_gray, face)
            # mask_landmarks_align(ref_image,,, mask_image,,,)
            landmarks_points = []
            landmarks_points = tuple(points)
            pt = []
            z = 0
            for idx in range(68):
                if (idx > 16) and (idx != 27) and (idx != 28):
                    continue
                if points[idx][0] < x1 :
                    x1 = points[idx][0] +1
                if points[idx][0] > x2 :
                    x2 = points[idx][0]
                if points[idx][1] < y1 :
                    y1 = points[idx][1] +1
                if points[idx][1] > y2 :
                    y2 = points[idx][1]
                if idx > 3 and idx < 14:
                    pt.append([points[idx][0],points[idx][1]])

        if(flag == True):
            rect = (x1, y1, x2-x1, y2-y1)
            img_copy = img.copy()
            #cv2.rectangle(img_copy, rect[:2], rect[2:], (0, 255, 0), 3)
            mask = np.zeros(img.shape[:2], np.uint8)
            grabcut_output = grabcut(img, mask, rect)
            grabcut_output = make_binary_image(grabcut_output)
            ## cv2.imwrite("grabcut_output/" + filename + ".jpg", grabcut_output)
            #print(points.mean(axis=0).astype("int"))
            #print(np.mean(pt, axis=0, dtype = np.int32))

            seeds_list = np.mean(pt, axis=0, dtype = np.int32)
            #red_color = (0, 0, 255) # BGR
            #cv2.circle(img_copy, seeds_list, 5, red_color, -1)
            seeds = [Point(seeds_list[1], seeds_list[0])]
            binaryImg = regionGrow(img, seeds, 35)
            kernel = np.ones((5,5), np.uint8)
            dilation_img = cv2.dilate(binaryImg, kernel, iterations = 1)
            # binaryImg = Image.fromarray(np.uint8(binaryImg * 255) , 'L')
            ## cv2.imwrite("regiongrow_output/" + filename + ".jpg", dilation_img)

            combImg = np.zeros(img.shape)
            height, weight = img.shape[:2]
            label = (255, 255, 255)
            for i in range(height):
                for j in range(weight):
                    #print(np.mean(grabcut_output[i][j]))
                    x = np.mean(dilation_img[i][j], dtype = int)
                    y = np.mean(grabcut_output[i][j], dtype = int)
                    if x == 255 and y == 255 :
                        combImg[i][j] = label
            cv2.imwrite(output_path, combImg)
            return output_path
            # cv2.imshow('combine result', combImg)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()