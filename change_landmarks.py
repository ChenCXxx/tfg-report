import cv2
import numpy as np
import dlib
import time
import os
import pathlib
from make_landmarks import MakeLandmarks
from matplotlib import pyplot as plt

flag = 0
def show(img):
    #轉換通道
    img_ = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img_)
def grabcut(img, mask, rect, iters=20):
    img_ = img.copy()
    bg_model = np.zeros((1, 65),np.float64)
    fg_model = np.zeros((1, 65),np.float64)
    cv2.grabCut(img.copy(), mask, rect, bg_model, fg_model, iters, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img_ = img*mask2[:,:,np.newaxis]
    return img_
def resize_img(img):
    scale_percent = 128/592 # percent of original size
    width = int(img.shape[1] * scale_percent)
    height = int(img.shape[0] * scale_percent)
    dim = (width, height)
    resize_img = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)  
        
    return resize_img

array_of_img1 = [] # this if for store all of the source image data
array_of_img2 = [] # this if for store all of the target image data
filename = []

img_file = "0303/swap" # output_filename
mask_file = "0303/mask" # output_maskname

source_path = "0303/source"
target_path = "0303/masked"

files = os.listdir(source_path) 

def Tgetavgstd(image, y, x, h, w):
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
            if i+40 > h/2 :
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

array_of_img1 = read_directory(source_path, array_of_img1)
array_of_img2 = read_directory(target_path, array_of_img2)

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

for x in range(len(array_of_img1 )):
    newName = files[x][:-4]
    img = array_of_img1[x]
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = np.zeros_like(img_gray)
    img2 = array_of_img2[x]
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    height, width, channels = img2.shape
    img2_new_face = np.zeros((height, width, channels), np.uint8)




    # Face 1
    faces = detector(img_gray)
    flag = 1
    for face in faces:
        flag = 0
        landmarks = predictor(img_gray, face)
        ML = MakeLandmarks(predictor)
        landmarks_points = []
        for n in range(0, 68):
            if n==0 or n==16:
                continue
            elif n>=17 and n<=27:
                continue
            elif n>=36 and n<=47:
                continue
            else:
                x = landmarks.part(n).x
                y = landmarks.part(n).y
                landmarks_points.append((x, y))
            



        points = np.array(landmarks_points, np.int32)
        convexhull = cv2.convexHull(points)
        # cv2.polylines(img, [convexhull], True, (255, 0, 0), 3)
        cv2.fillConvexPoly(mask, convexhull, 255)

        face_image_1 = cv2.bitwise_and(img, img, mask=mask)

        # Delaunay triangulation
        rect = cv2.boundingRect(convexhull)
        subdiv = cv2.Subdiv2D(rect)
        subdiv.insert(landmarks_points)
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
                indexes_triangles.append(triangle)

    if flag == 1 :
        continue

    # Face 2
    faces2 = detector(img2_gray)
    chk = 0
    flag = 1
    for x in range(len(faces2)):
        flag = 0
        face = faces[x]
        face2 = faces2[x]
        pt2 = ML.make_landmarks_align(img, img_gray, face, img2, img2_gray, face2)
        landmarks_points2 = []
        nn = 0
        fx, fy, ex, ey = img2.shape[1], img2.shape[0], 0, 0
        print(fx, fy, ex, ey)
        p_num = 0
        for n in range(68):
            if n==0 or n==16:
                continue
            elif n>=17 and n<=27:
                continue
            elif n>=36 and n<=47:
                continue
            else:
                x = pt2[n][0]
                y = pt2[n][1]
                landmarks_points2.append((x, y))
                p_num += 1
                fx = min(fx, x)
                fy = min(fy, y)
                ex = max(ex, x+5)
                ey = max(ey, y+5)

        
        # landmarks = predictor(img2_gray, face)
        points2 = []
        points2 = np.array(landmarks_points2, np.int32)
        img2_copy = img2.copy()
        for idx in range(p_num):
            pos = (points2[idx, 0], points2[idx, 1])
            cv2.circle(img2_copy, pos, 3, (0, 0, 255), -1)
        # test
        '''
        cv2.namedWindow("circle_landmarks", cv2.WINDOW_NORMAL) 
        cv2.imshow("circle_landmarks", img2_copy)
        cv2.waitKey(0) 
        '''
        '''
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmarks_points2.append((x, y))
            '''
        
        points2 = np.array(landmarks_points2, np.int32)
        convexhull2 = cv2.convexHull(points2)

    if flag == 1 :
        continue
    
    lines_space_mask = np.zeros_like(img_gray)
    lines_space_new_face = np.zeros_like(img2)
    # Triangulation of both faces
    for triangle_index in indexes_triangles:
        # Triangulation of the first face
        tr1_pt1 = landmarks_points[triangle_index[0]]
        tr1_pt2 = landmarks_points[triangle_index[1]]
        tr1_pt3 = landmarks_points[triangle_index[2]]
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
        tr2_pt1 = landmarks_points2[triangle_index[0]]
        tr2_pt2 = landmarks_points2[triangle_index[1]]
        tr2_pt3 = landmarks_points2[triangle_index[2]]
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
    
    x1 = face2.left()
    y1 = face2.top()
    x2 = face2.right()
    y2 = face2.bottom()
    y = y1-3 if y1-10 >= 0 else y1
    x = x1-3 if x1-10 >= 0 else x1
    h = y2-y1+50 if y2+20 < 592 else 592-y2
    w = x2-x1+50 if x2+20 < 592 else 592-x2
    img2_new_face_LAB = cv2.cvtColor(img2_new_face,cv2.COLOR_BGR2LAB)
    img2_LAB = original = cv2.cvtColor(img2,cv2.COLOR_BGR2LAB)
    source_avg,source_std = Sgetavgstd(img2_new_face_LAB, y, x, h, w)
    target_avg,target_std = Tgetavgstd(img2_LAB, y, x, h, w)

    # Face swapped (putting 1st face into 2nd face)
    img2_face_mask = np.zeros_like(img2_gray)
    img2_head_mask = cv2.fillConvexPoly(img2_face_mask, convexhull2, 255)
    img2_face_mask = cv2.bitwise_not(img2_head_mask)


    img2_head_noface = cv2.bitwise_and(img2, img2, mask=img2_face_mask)
    result = cv2.add(img2_head_noface, img2_new_face)

    (x, y, w, h) = cv2.boundingRect(convexhull2)
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
                color_trans[y+i,x+j,k] = LAB[y+i,x+j,k]
                
    #cv2.imshow("after", color_trans)
    #cv2.waitKey(0)
    # seamlessclone = cv2.seamlessClone(color_trans, img2, img2_head_mask, center_face2, cv2.NORMAL_CLONE)
    
    

    path_list = [img_file, newName + "." + "jpg"]
    head = '' # synthetic image path
    for path in path_list:
        head = os.path.join(head, path)
    color_trans_128 = resize_img(color_trans)
    cv2.imwrite(head, color_trans_128)


    '''
    path2_list = [img_file + "_off", newName + "." + "jpg"]
    head2 = '' # synthetic image path
    for path2 in path2_list:
        head2 = os.path.join(head2, path2)
    cv2.imwrite(head2, result)
    '''

    # print(fx, fy, ex, ey)
    mask = np.zeros(img2.shape[:2], np.uint8)
    rect = (fx, fy, ex-fx, ey-fy)
    img2_copy = img2.copy()

    cv2.rectangle(img2_copy, rect[:2], rect [2:], (0, 255, 0), 3)
    show(img2_copy)
    img2_GC = grabcut(img2, mask, rect)
    # print(img2_head_mask)
    # plt.figure(1)
    # show(img2_GC)

    # plt.figure(2)
    # show(img2_head_mask)

    binary_mask = []
    image_height, image_width = img2.shape[:2]
    binary_mask = np.zeros((image_height, image_width, 3), np.uint8)
    
    
    
    for xx in range(image_width):
        for yy in range(image_height):
            # print(img2_GC[xx][yy])
            if img2_GC[xx, yy].all() != 0 and img2_head_mask[xx, yy].all() == 0:
                binary_mask[xx, yy, 0] = 255
                binary_mask[xx, yy, 1] = 255
                binary_mask[xx, yy, 2] = 255
    
    binary_mask_128 = resize_img(binary_mask)
    path2_list = [mask_file , newName + "." + "jpg"]
    mask_path = '' # synthetic image path
    for path2 in path2_list:
        mask_path = os.path.join(mask_path, path2)
    cv2.imwrite(mask_path, binary_mask_128)

    show(binary_mask_128)
    plt.show()

    str_p = "Success: " + head
    print(str_p)

    





    #cv2.imshow("seamlessclone", seamlessclone)
    #cv2.waitKey(0)
    # #\cv2.destroyAllWindows()