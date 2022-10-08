
# 0604explain_landmarks_align.py

from collections import OrderedDict
import numpy as np
import cv2
import dlib

#創建一個tuple("部位", (point_start, point_end))
FACIAL_LANDMARKS_68_IDXS = OrderedDict([
	("mouth", (48, 68)),
	("inner_mouth", (60, 68)),
	("right_eyebrow", (17, 22)),
	("left_eyebrow", (22, 27)),
	("right_eye", (36, 42)),
	("left_eye", (42, 48)),
	("nose", (27, 36)),
	("jaw", (0, 17))
])

def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	# shape.num_part = point
	coords = np.zeros([shape.num_parts, 2], dtype=dtype)

	# loop over all facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, shape.num_parts):
		coords[i] = (shape.part(i).x, shape.part(i).y)

	# return the list of (x, y)-coordinates
	return coords

def make_landmarks_align(img, img_gray, rect, img2, img2_gray, rect2):
		# for face 1 -> source
		shape = predictor(img_gray, rect)

		shape = shape_to_np(shape)

		# extract the left and right eye (x, y)-coordinates
		(lStart, lEnd) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
		(rStart, rEnd) = FACIAL_LANDMARKS_68_IDXS["right_eye"]

		leftEyePts = shape[lStart:lEnd]
		rightEyePts = shape[rStart:rEnd]

		# compute the center of mass for each eye
		# mean(axis=0) 取縱行的平均值 / (axis = 1) 取橫行的平均值
		leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
		rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

		# compute the angle between the eye centroids
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]
		angle = np.degrees(np.arctan2(dY, dX)) - 180
		eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2.0,
			(leftEyeCenter[1] + rightEyeCenter[1]) // 2.0)
		dist = np.sqrt((dX ** 2) + (dY ** 2))

		# for face 2 -> target face
		shape2 = predictor(img2_gray, rect2)
		shape2 = shape_to_np(shape2)

		(lStart2, lEnd2) = FACIAL_LANDMARKS_68_IDXS["left_eye"]
		(rStart2, rEnd2) = FACIAL_LANDMARKS_68_IDXS["right_eye"]
		leftEyePts2 = shape2[lStart2:lEnd2]
		rightEyePts2 = shape2[rStart2:rEnd2]

		leftEyeCenter2 = leftEyePts2.mean(axis=0).astype("int")
		rightEyeCenter2 = rightEyePts2.mean(axis=0).astype("int")
		dY2 = rightEyeCenter2[1] - leftEyeCenter2[1]
		dX2 = rightEyeCenter2[0] - leftEyeCenter2[0]
		angle2 = np.degrees(np.arctan2(dY2, dX2)) - 180
		eyesCenter2 = ((leftEyeCenter2[0] + rightEyeCenter2[0]) // 2.0,
			(leftEyeCenter2[1] + rightEyeCenter2[1]) // 2.0)
		dist2 = np.sqrt((dX2 ** 2) + (dY2 ** 2))
		scale = float(dist2) / float(dist)
		# center2 - center1
		center_err_x = (eyesCenter2[0] - eyesCenter[0]) * scale
		center_err_y = (eyesCenter2[1] - eyesCenter[1]) * scale

		new_landmarks = np.zeros([68, 2], dtype="int")
		for ind in range(0, 68):
			new_landmarks[ind] = ((eyesCenter2[0] + (shape[ind][0] - eyesCenter[0])*dist2/dist)
				, (eyesCenter2[1] + (shape[ind][1] - eyesCenter[1])*dist2/dist))
		
		#eyesCenter2 = list(eyesCenter2)
		#eyesCenter2 = [int(x) for x in eyesCenter2]
		#return eyesCenter2
		return new_landmarks


file = "0604_landmarks"
unmasked_path = "0604_landmarks\image_0226.jpg"
masked_path = "0604_landmarks\image_0217_cloth.jpg"

unmasked_img = cv2.imread(unmasked_path)
unmasked_img_gray = cv2.cvtColor(unmasked_img, cv2.COLOR_BGR2GRAY)
masked_img = cv2.imread(masked_path)
masked_img_gray = cv2.cvtColor(masked_img, cv2.COLOR_BGR2GRAY)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

faces = detector(unmasked_img_gray)
faces2 = detector(masked_img_gray)

landmarks = predictor(unmasked_img_gray, faces[0])
landmarks_points =[]
for n in range(0, 68):
    x = landmarks.part(n).x
    y = landmarks.part(n).y
    landmarks_points.append((x, y))
landmarks_points = np.array(landmarks_points, np.int32)

print(faces2[0])
landmarks2 = predictor(masked_img_gray, faces2[0])
landmarks_points2 =[]
for n in range(0, 68):
    x = landmarks2.part(n).x
    y = landmarks2.part(n).y
    landmarks_points2.append((x, y))
landmarks_points2 = np.array(landmarks_points2, np.int32)


landmarks_points2_align = make_landmarks_align(unmasked_img, unmasked_img_gray, faces[0], 
masked_img, masked_img_gray, faces2[0])

unmasked_img_copy = unmasked_img.copy()
masked_img_copy = masked_img.copy()
masked_img_align_copy = masked_img.copy()
circle_size = 3
color = (0, 0, 255)
for i in range(68):
        cv2.circle(unmasked_img_copy, landmarks_points[i], circle_size, color, -1)

for i in range(68):
        cv2.circle(masked_img_copy, landmarks_points2[i], circle_size, color, -1)

for i in range(68):
        cv2.circle(masked_img_align_copy, landmarks_points2_align[i], circle_size, color, -1)

cv2.imshow("unmasked_img", unmasked_img_copy)
cv2.waitKey(0)
cv2.imshow("masked_img_copy", masked_img_copy)
cv2.waitKey(0)
cv2.imshow("masked_img_align_copy", masked_img_align_copy)
cv2.waitKey(0)
cv2.destroyAllWindows()

folder = "0604_landmarks"
cv2.imwrite( folder + "\\" + "unmasked_img.jpg" , unmasked_img_copy)
cv2.imwrite( folder + "\\" + "masked_img.jpg" , masked_img_copy)
cv2.imwrite( folder + "\\" + "masked_img_align.jpg", masked_img_align_copy)
