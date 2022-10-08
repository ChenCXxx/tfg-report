import cv2
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

unmasked_path = [] # this if for store all of the source image data
masked_path = [] # this if for store all of the target image data
filename = []

output_file = "0606substract\\output" # output_filename
source_path = "0606substract\\unmasked"
target_path = "0606substract\\masked"
binary_mask_path = "0606substract\\binary"
diff_set_file = "0606substract\\diff"

all_file = "0606substract\\592\\"
all_resize_file = "0606substract\\128\\"

QQ = "0283"

def resize_img(img):

    scale_percent = 12800/592 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    resize_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)  
        
    return resize_img

def Tgetavgstd(image, height, width, landmarks_mask, mask, xx, yy):
    igg = image.copy()
    #cv2.circle(igg, [xx, yy-10], 4, (255, 255, 255), -1)
    #cv2.circle(igg, [xx, yy], 4, (255, 255, 255), -1)
    #plt.imshow(igg)
    #plt.show()
    
    #得到均值和标准差
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
            if landmarks_mask[i, j] == 0 :
                continue
            if mask[i, j] != 0 :
                continue
            grayDiff = abs((image[i, j]).astype(int) - (image[xx, yy-25]).astype(int)) 
            grayDiff = grayDiff.mean(axis=0).astype("int")
            #print(grayDiff)
            if grayDiff > 30:
                continue
            num = num+1
            image_avg_l += image[i,j,0]
            image_std_l.append(image[i,j,0])
            image_avg_a += image[i,j,1]
            image_std_a.append(image[i,j,1])
            image_avg_b += image[i,j,2]
            image_std_b.append(image[i,j,2])
    image_avg_l = image_avg_l/num
    image_avg_a = image_avg_a/num
    image_avg_b = image_avg_b/num
    std_l = np.std(image_std_l, ddof=1)
    std_a = np.std(image_std_a, ddof=1)
    std_b = np.std(image_std_b, ddof=1)
    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(std_l)
    std.append(std_a)
    std.append(std_b)
    return (avg,std)

def Sgetavgstd(image, y, x, h, w):
    #得到均值和标准差
    avg = []
    std = []
    image_avg_l = 0
    image_avg_a = 0
    image_avg_b = 0
    image_std_l = []
    image_std_a = []
    image_std_b = []
    num = 0
    for i in range(h) :
        for j in range(w) :
            if img2_new_face[y+i, x+j, 0] == 0 and img2_new_face[y + i, x + j, 1] == 0 and img2_new_face[y + i, x + j, 1] == 0:
                continue
            num = num+1
            image_avg_l += image[y+i,x+j,0]
            image_std_l.append(image[y+i,x+j,0])
            image_avg_a += image[y+i,x+j,1]
            image_std_a.append(image[y+i,x+j,1])
            image_avg_b += image[y+i,x+j,2]
            image_std_b.append(image[y+i,x+j,2])
    image_avg_l = image_avg_l/num
    image_avg_a = image_avg_a/num
    image_avg_b = image_avg_b/num
    std_l = np.std(image_std_l, ddof=1)
    std_a = np.std(image_std_a, ddof=1)
    std_b = np.std(image_std_b, ddof=1)
    avg.append(image_avg_l)
    avg.append(image_avg_a)
    avg.append(image_avg_b)
    std.append(std_l)
    std.append(std_a)
    std.append(std_b)
    return (avg,std)

def read_directory(directory_name, array_of_img):
    # this loop is for read each image in this foder,directory_name is the foder name with images.
    for filename in os.listdir(r"./"+directory_name):
        #print(filename) #just for test
        #img is used to store the image data 
        img = cv2.imread(directory_name + "/" + filename)
        array_of_img.append(img)
    return array_of_img

def dir_array(dir_path):
    itemlist = os.listdir(dir_path)
    dir_list = []
    # 獲取目錄檔案列表
    for item in itemlist:
        # 連線成完整路徑
        item_path = os.path.join(dir_path, item)
        dir_list.append(item_path) 
    
    return dir_list

unmasked_path = dir_array(source_path)
masked_path = dir_array(target_path)

def extract_index_nparray(nparray):
    index = None
    for num in nparray[0]:
        index = num
        break
    return index

'''
img = cv2.imread("test_data/source/image_0002.jpg")
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
mask = np.zeros_like(img_gray)
img2 = cv2.imread("test_data/target/image_0001_N95.jpg")
img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
'''

MB = mask_binary()
MP = Mask_points()

for x in range(len(unmasked_path)):
    newName = unmasked_path[x][-8:-4]
    # QQ = masked_path[x][-8:-4]
    print(str(QQ))
    outputName = output_file + "/" + str(QQ) + "_output" + ".jpg"
    noTransName = output_file + "/" + str(QQ) + "_noTrans" + ".jpg"
    diffsetName = diff_set_file + "/" + str(QQ) + ".jpg"
    img = cv2.imread(unmasked_path[x])
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    img2 = cv2.imread(masked_path[x])
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    mask2 = np.zeros_like(img2_gray)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    height, width, channels = img2.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)

    faces = detector(img_gray)
    chk = 1
    if (faces):
        chk = 0
    if chk == 1 :
        continue

    binary_path = MB.find_mask_binary(masked_path[x], unmasked_path[x], binary_mask_path, newName)
    print(binary_path)
    if(binary_path):
        UnMaskedPt, MaskedPt = MP.mask_points(binary_path, masked_path[x], unmasked_path[x])
    else :
        continue
    binary_mask = cv2.imread(binary_path)
    
    # Face 1 -> unmasked_img
    faces = detector(img_gray)
    flag = 1
    ML = MakeLandmarks(predictor)
    for face in faces:
        flag = 0

        landmarks = predictor(img_gray, face)
        landmarks_points =[]
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points.append((x, y))
        landmarks_points = np.array(landmarks_points, np.int32)
        ch = cv2.convexHull(landmarks_points)
        landmarks_mask = np.zeros_like(img_gray)
        cv2.fillConvexPoly(landmarks_mask, ch, 255)

        mask_points = []
        for j in range(len(UnMaskedPt)):
            mask_points.append((int(UnMaskedPt[j][0]), int(UnMaskedPt[j][1])))

        unmasked_img_copy = img.copy()
        yy, xmax, xmin = 592, 0, 592
        for i in range(len(mask_points)):
            cv2.circle(unmasked_img_copy, mask_points[i], 3, (0, 0, 255), -1)
        for i in range(68):
            cv2.circle(unmasked_img_copy, landmarks_points[i], 3, (0, 255, 0), -1)
        # cv2.imshow("unmasked_img_copy", unmasked_img_copy)
        # cv2.waitKey(0)

        # 暫時刪掉 points -> 得牢內使用的點 mask_convexhull -> 選取範圍
        points = np.array(mask_points, np.int32)
        mask_convexhull = cv2.convexHull(points)
        # cv2.polylines(img, [mask_convexhull], True, (255, 0, 0), 3)
        cv2.fillConvexPoly(mask, mask_convexhull, 255)

        and_1 = cv2.bitwise_and(landmarks_mask, mask)
        #print(landmarks_points)
        #print(points)
        comb_points = []
        #cv2.imshow("and_1", and_1)
        #cv2.waitKey(0)
        a1, b1 = 0, 0
        landmarks_list = []
        for i in range(len(landmarks_points)):
            if i > 36 or i < 28 :
                if(and_1[landmarks_points[i][1]][landmarks_points[i][0]] == 255) :
                    a1 += 1
                    landmarks_list.append(i)
        mask_list = []
        for i in range(len(points)):
            if(and_1[points[i][1]][points[i][0]] == 255) :
                b1 += 1
                mask_list.append(i)
                # comb_points.append((int(points[i][0]), int(points[i][1])))



        """# Delaunay triangulation
        rect = cv2.boundingRect(mask_convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(mask_points)
        triangles = subdiv.getTriangleList()
        triangles = np.array(triangles, dtype=np.int32)

        indexes_triangles = []
        for t in triangles:
            pt1 = (t[0], t[1])
            pt2 = (t[2], t[3])
            pt3 = (t[4], t[5])


            index_pt1 = np.where((points == pt1).all(axis=1))
            index_pt1 = extract_index_nparray(index_pt1)

            index_pt2 = np.where((points == pt2).all(axis=1))
            index_pt2 = extract_index_nparray(index_pt2)

            index_pt3 = np.where((points == pt3).all(axis=1))
            index_pt3 = extract_index_nparray(index_pt3)

            if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
                triangle = [index_pt1, index_pt2, index_pt3]
                indexes_triangles.append(triangle)"""

    if flag == 1 :
        continue

    # Face 2 -> masked_img
    faces2 = detector(img2_gray)
    chk = 0
    flag = 1
    for x in range(len(faces2)):
        flag = 0
        face = faces[x]
        face2 = faces2[x]
        landmarks_points2 = []
        landmarks_points2 = ML.make_landmarks_align(img, img_gray, face, img2, img2_gray, face2)
        # landmarks = predictor(img2_gray, face)
        ch2 = cv2.convexHull(landmarks_points2)
        landmarks_mask2 = np.zeros_like(img2_gray)
        cv2.fillConvexPoly(landmarks_mask2, ch2, 255)


        # landmarks = predictor(img2_gray, face)
        mask_points2 = []
        for j in range(len(MaskedPt)):
            mask_points2.append((int(MaskedPt[j][0]), int(MaskedPt[j][1])))

        '''
        mask_points2 = tuple(points2)
        img2_copy = img2.copy()
        for idx in range(68):
            pos = (points2[idx, 0], points2[idx, 1])
            cv2.circle(img2_copy, pos, 3, (0, 0, 255), -1)
            '''

        ppoints2 = np.array(mask_points2, np.int32)
        mask_convexhull2 = cv2.convexHull(ppoints2)
        cv2.fillConvexPoly(mask2, mask_convexhull2, 255)
        masked_img_copy = img2.copy()
        
        #交集的遮罩and_2
        and_2 = cv2.bitwise_and(landmarks_mask2, mask2)
        comb_points2 = []
        a2, b2 = 0, 0
        #把人臉特徵點以及口罩特徵點在遮罩內的點挑出來放在comb_point2
        for i in range(len(landmarks_list)):
            if i > 36 or i < 28 :
                if(and_2[landmarks_points2[landmarks_list[i]][1]][landmarks_points2[landmarks_list[i]][0]] == 255) :
                    a2 += 1
                    comb_points.append((int(landmarks_points[landmarks_list[i]][0]), int(landmarks_points[landmarks_list[i]][1])))
                    comb_points2.append((int(landmarks_points2[landmarks_list[i]][0]), int(landmarks_points2[landmarks_list[i]][1])))
        comb_img_copy = img.copy()
        for i in range(len(comb_points)):
            cv2.circle(comb_img_copy, comb_points[i], 3, (0, 0, 255), -1)
        # cv2.imshow("comb_img_copy", comb_img_copy)
        # cv2.waitKey(0)
        comb_img_copy2 = img2.copy()
        for i in range(len(comb_points2)):
            cv2.circle(comb_img_copy2, comb_points2[i], 3, (0, 0, 255), -1)
        for i in range(len(mask_list)):
            if(b2 < b1 and and_2[ppoints2[mask_list[i]][1]][ppoints2[mask_list[i]][0]] == 255) :
                b2 += 1
                comb_points.append((int(points[mask_list[i]][0]), int(points[mask_list[i]][1])))
                comb_points2.append((int(ppoints2[mask_list[i]][0]), int(ppoints2[mask_list[i]][1])))
        # cv2.imshow("comb_img_copy", comb_img_copy)
        # cv2.waitKey(0)
        # cv2.imshow("comb_img_copy2", comb_img_copy2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        if b2 < b1:
            b1 = b2
        
        bb = 0
        '''for i in range(len(points)):
            if(bb < b1 and and_1[points[i][1]][points[i][0]] == 255) :
                bb += 1
                comb_points.append((int(points[i][0]), int(points[i][1])))'''

        # 測試用
        comb_img_copy = img.copy()
        for i in range(len(comb_points)):
            cv2.circle(comb_img_copy, comb_points[i], 3, (0, 0, 255), -1)
        # cv2.imshow("comb_img_copy", comb_img_copy)
        # cv2.waitKey(0)
        comb_img_copy2 = img2.copy()
        for i in range(len(comb_points2)):
            cv2.circle(comb_img_copy2, comb_points2[i], 3, (0, 0, 255), -1)
        print(a1, b1, a2, b2)
        # cv2.imshow("comb_img_copy2", comb_img_copy2)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

        diff_rem = cv2.subtract(landmarks_mask2, mask2) 
        diff_set = cv2.subtract(mask2, landmarks_mask2) 
        #diff_set = cv2.bitwise_xor(landmarks_mask2, mask2)
        cv2.imwrite(diffsetName, diff_set)

        comb_points_arr = np.array(comb_points, np.int32)
        comb_convexhull = cv2.convexHull(comb_points_arr)

        comb_points_arr2 = np.array(comb_points2, np.int32)
        mask_convexhull2 = cv2.convexHull(comb_points_arr2)

        masked_img_copy = img2.copy()
        for i in range(len(mask_points2)):
            cv2.circle(masked_img_copy, (mask_points2[i][0], mask_points2[i][1]), 3, (0, 0, 255), -1)
            yy = min(yy, mask_points2[i][1])
            xmin = min(xmin, mask_points2[i][0])
            xmax = max(xmax, mask_points2[i][0])
        xx = (xmin + xmax) / 2 
        print(xmin)
        print(xmax)
        print(yy)
        xx = int(xx)
        for i in range(68):
            cv2.circle(masked_img_copy, (landmarks_points2[i][0], landmarks_points2[i][1]), 3, (0, 255, 0), -1)
        # cv2.imshow("masked_img_copy", masked_img_copy)
        # cv2.imwrite("0416-test\MaskPointImg.jpg", masked_img_copy)

    # Delaunay triangulation
    rect = cv2.boundingRect(comb_convexhull)
    subdiv = cv2.Subdiv2D(rect)
    subdiv.insert(comb_points)
    triangles = subdiv.getTriangleList()
    triangles = np.array(triangles, dtype=np.int32)

    indexes_triangles = []
    for t in triangles:
        pt1 = (t[0], t[1])
        pt2 = (t[2], t[3])
        pt3 = (t[4], t[5])


        index_pt1 = np.where((comb_points_arr == pt1).all(axis=1))
        index_pt1 = extract_index_nparray(index_pt1)

        index_pt2 = np.where((comb_points_arr == pt2).all(axis=1))
        index_pt2 = extract_index_nparray(index_pt2)

        index_pt3 = np.where((comb_points_arr == pt3).all(axis=1))
        index_pt3 = extract_index_nparray(index_pt3)

        if index_pt1 is not None and index_pt2 is not None and index_pt3 is not None:
            triangle = [index_pt1, index_pt2, index_pt3]
            indexes_triangles.append(triangle)

    
    if flag == 1 :
        continue
    
    lines_space_mask = np.zeros_like(img_gray) 
    lines_space_new_face = np.zeros_like(img2)
    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt1 = comb_points[triangle_index[0]]
        tr1_pt2 = comb_points[triangle_index[1]]
        tr1_pt3 = comb_points[triangle_index[2]]
        triangle1 = np.array([tr1_pt1, tr1_pt2, tr1_pt3], np.int32)

        rect1 = cv2.boundingRect(triangle1)
        (x, y, w, h) = rect1
        cropped_triangle = img[y: y + h, x: x + w]
        cropped_tr1_mask = np.zeros((h, w), np.uint8)

        points = np.array([[tr1_pt1[0] - x, tr1_pt1[1] - y],
                        [tr1_pt2[0] - x, tr1_pt2[1] - y],
                        [tr1_pt3[0] - x, tr1_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr1_mask, points, 255)

        # Lines space
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt2, 255)
        cv2.line(lines_space_mask, tr1_pt2, tr1_pt3, 255)
        cv2.line(lines_space_mask, tr1_pt1, tr1_pt3, 255)
        lines_space = cv2.bitwise_and(img, img, mask=lines_space_mask)

        # Triangulation of second face
        tr2_pt1 = comb_points2[triangle_index[0]]
        tr2_pt2 = comb_points2[triangle_index[1]]
        tr2_pt3 = comb_points2[triangle_index[2]]
        triangle2 = np.array([tr2_pt1, tr2_pt2, tr2_pt3], np.int32)

        rect2 = cv2.boundingRect(triangle2)
        (x, y, w, h) = rect2

        cropped_tr2_mask = np.zeros((h, w), np.uint8)

        points2 = np.array([[tr2_pt1[0] - x, tr2_pt1[1] - y],
                            [tr2_pt2[0] - x, tr2_pt2[1] - y],
                            [tr2_pt3[0] - x, tr2_pt3[1] - y]], np.int32)

        cv2.fillConvexPoly(cropped_tr2_mask, points2, 255)

        # Warp triangles
        points = np.float32(points)
        points2 = np.float32(points2)
        M = cv2.getAffineTransform(points, points2)
        warped_triangle = cv2.warpAffine(cropped_triangle, M, (w, h))
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=cropped_tr2_mask)

        # Reconstructing destination face
        img2_new_face_rect_area = img2_new_face[y: y + h, x: x + w]
        img2_new_face_rect_area_gray = cv2.cvtColor(img2_new_face_rect_area, cv2.COLOR_BGR2GRAY)
        _, mask_triangles_designed = cv2.threshold(img2_new_face_rect_area_gray, 1, 255, cv2.THRESH_BINARY_INV)
        warped_triangle = cv2.bitwise_and(warped_triangle, warped_triangle, mask=mask_triangles_designed)

        img2_new_face_rect_area = cv2.add(img2_new_face_rect_area, warped_triangle)
        img2_new_face[y: y + h, x: x + w] = img2_new_face_rect_area
    
    # cv2.imshow("img2_new_face", img2_new_face)
    # cv2.waitKey(0)

    # print(img2_new_face)
    img2_new_face_copy = img2_new_face.copy()
    for i in range(height) :
        for j in range(width) :
            if img2_new_face_copy[i][j].all() != 0 :
                img2_new_face_copy[i][j] = 255
            else :
                img2_new_face_copy[i][j] = 0
    img2_new_face_copy = cv2.bitwise_not(img2_new_face_copy)
    img2_new_face_copy = cv2.cvtColor(img2_new_face_copy, cv2.COLOR_RGB2GRAY)
    # img2_new_face_copy = img2_new_face_copy.convert('L')
    # plt.imshow(img2_new_face_copy)
    # plt.show()

    x1 = face2.left()
    y1 = face2.top()
    x2 = face2.right()
    y2 = face2.bottom()
    y = y1-5 if y1-10 >= 0 else y1
    x = x1-5 if x1-10 >= 0 else x1
    h = y2-y1+50 if y2+20 < 592 else 592-y2
    w = x2-x1+50 if x2+20 < 592 else 592-x2

    img2_new_face_LAB = cv2.cvtColor(img2_new_face,cv2.COLOR_BGR2LAB)
    img2_LAB = original = cv2.cvtColor(img2,cv2.COLOR_BGR2LAB)

    # 目標圖片利用參考圖片去更改顏色
    source_avg,source_std = Sgetavgstd(img2_new_face_LAB, y, x, h, w) # 目標(無口罩(去對應口罩部分)圖片)
    target_avg,target_std = Tgetavgstd(img2_LAB, height, width, landmarks_mask2, mask2, xx, yy)   # 參考(有口罩(只取上半部臉)圖片)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                 

    # Face swapped (putting 1st face into 2nd face)
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, mask_convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)
    '''plt.imshow(img2_face_mask)
    plt.show()'''
    
    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_new_face_copy)
    result = cv2.add(img2_head_noface, img2_new_face)

    (x, y, w, h) = cv2.boundingRect(mask_convexhull2)
    center_face2 = (int((x + x + w) / 2), int((y + y + h) / 2))
    '''
    seamlessclone = cv2.seamlessClone(result, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

    os.makedirs(img_file, exist_ok=True)

    path_list = [img_file, newName + "_off" + "." + "jpg"]
    head = '' # synthetic image path
    for path in path_list:
        head = os.path.join(head, path)
    
    cv2.imwrite(head, seamlessclone)

    '''

    #cv2.imshow("before", seamlessclone)
    #cv2.waitKey(0)
    color_trans = result.copy()
    color_trans2 = result.copy()
    LAB = np.zeros_like(result)
    for i in range(0,h):
        for j in range(0,w):
            if img2_new_face[y+i, x+j, 0] == 0 and img2_new_face[y + i, x + j, 1] == 0 and img2_new_face[y + i, x + j, 2] == 0:
                continue
            for k in range(0,3):
                t = img2_new_face_LAB[y+i,x+j,k]
                t = (t-source_avg[k])*(target_std[k]/source_std[k]) + target_avg[k]
                t = 0 if t<0 else t
                t = 255 if t>255 else t
                LAB[y+i,x+j,k] = t
    LAB = cv2.cvtColor(LAB, cv2.COLOR_LAB2BGR)
    for i in range(0,h):
        for j in range(0,w):
            if img2_new_face[y+i, x+j, 0] == 0 and img2_new_face[y + i, x + j, 1] == 0 and img2_new_face[y + i, x + j, 2] == 0:
                continue
            for k in range(0,3):
                color_trans[y+i,x+j,k] = LAB[y+i,x+j,k]*0.5 + color_trans[y+i,x+j,k]*0.5
    
    img2_change = color_trans.copy()
    #cv2.imshow("color_trans", color_trans)
    #cv2.imshow("color_trans2", color_trans2)
    img2_change = cv2.cvtColor(img2_change, cv2.COLOR_BGR2LAB)
    #cv2.fillConvexPoly(img2_change, mask_convexhull2, target_avg)
    img2_change = cv2.cvtColor(img2_change, cv2.COLOR_LAB2BGR)
    #cv2.imshow("img2_change", img2_change)
    
    seamlessclone = cv2.seamlessClone(img2_change, img2_change, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

    line_img = seamlessclone.copy()
    #line_img = cv2.polylines(MaskedPt)
    #cv2.imshow("seamlessclone", seamlessclone)
    #cv2.imshow("after", color_trans)
    
    
    #line_mask = seamlessclone.copy()
    line_mask = np.zeros_like(img2)
    line_mask.astype("uint8")
    ppoints2 = ppoints2.reshape((-1, 1, 2)) 
    cv2.polylines(line_mask, [mask_convexhull2], True, (255, 255, 255), 2)
    line_mask = cv2.cvtColor(line_mask, cv2.COLOR_BGR2GRAY)
    # line_img = cv2.inpaint(line_img, line_mask, 5,cv2.INPAINT_TELEA)
    #cv2.imshow("line_img",line_img)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
    # seamlessclone = cv2.seamlessClone(color_trans, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)

    cv2.imwrite(all_file + "masked_landmarks.jpg", landmarks_mask2)
    cv2.imwrite(all_file + "unmasked_img.jpg", img)
    cv2.imwrite(all_file + "masked_img.jpg", img2)
    cv2.imwrite(all_file + "diff_set.jpg", diff_set)
    cv2.imwrite(all_file + "noTrans.jpg", result)
    cv2.imwrite(all_file + "binary_mask.jpg", binary_mask)
    cv2.imwrite(all_file + "diff_rem.jpg", diff_rem)
    cv2.imwrite(all_file + "mask.jpg", mask2)

    cv2.imwrite(all_resize_file + "masked_landmarks.jpg", resize_img(landmarks_mask2))
    cv2.imwrite(all_resize_file + "unmasked_img.jpg", resize_img(img))
    cv2.imwrite(all_resize_file + "masked_img.jpg", resize_img(img2))
    cv2.imwrite(all_resize_file + "diff_set.jpg", resize_img(diff_set))
    cv2.imwrite(all_resize_file + "noTrans.jpg", resize_img(result))
    cv2.imwrite(all_resize_file + "binary_mask.jpg", resize_img(binary_mask))
    cv2.imwrite(all_resize_file + "diff_rem.jpg", resize_img(diff_rem))
    cv2.imwrite(all_resize_file + "mask.jpg", resize_img(mask2))
    '''
    for i in range(0,h):
        for j in range(0,w):
            if img2_new_face[y+i, x+j, 0] == 0 and img2_new_face[y + i, x + j, 1] == 0 and img2_new_face[y + i, x + j, 2] == 0:
                continue
            #if img2_new_face[y+i, x+j+20, 0] ==0 or img2_new_face[y+i, x+j-20, 0] ==0 or img2_new_face[y+i+20, x+j, 0] ==0 or img2_new_face[y+i-20, x+j, 0] ==0 or img2_new_face[y+i+20, x+j+20, 0] ==0 or img2_new_face[y+i-20, x+j-20, 0] ==0 :
            #    continue
            for k in range(0,3):
                color_trans[y+i,x+j,k] = seamlessclone[y+i,x+j,k]*0.5 + color_trans[y+i,x+j,k]*0.5
    #cv2.imshow("after", seamlessclone)
    #cv2.waitKey(0)
    '''

    '''
    path_list = [img_file, newName + "." + "jpg"]
    head = '' # synthetic image path
    for path in path_list:
        head = os.path.join(head, path)
    #cv2.imwrite(head, result)
    cv2.imwrite(head, color_trans)


    path2_list = [img_file + "_off", newName + "." + "jpg"]
    head2 = '' # synthetic image path
    for path2 in path2_list:
        head2 = os.path.join(head2, path2)
    cv2.imwrite(head2, result)

    
    
    str_p = "Success: " + head
    print(str_p)'''
