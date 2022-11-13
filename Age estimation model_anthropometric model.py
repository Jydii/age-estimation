from imutils import face_utils
import numpy as np
import imutils
import dlib
import cv2
import glob
import pickle
import os.path





def euclidean_dist(ptA, ptB):
	# compute and return the euclidean distance between the two
	# points
	return np.linalg.norm(ptA - ptB)

path = "images/aaf+fgnet_front/*.jpg"

# initialize dlib's face detector (HOG-based) and then create
# the facial landmark predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# List & Dict of Extracted Facial Features
data = []
face_dict = {}

for file in glob.glob(path): #iterate through each file in path
	# Image Processing
	# load the input image, resize it, and convert it to grayscale
	image = cv2.imread(file)
	resized = imutils.resize(image, width=500)
	gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)

	# Face Detection
	# Get frontal face in the grayscale image
	rects = detector(gray, 1)

	# loop over the face detections
	for (i,rect) in enumerate(rects):

		# determine the facial landmarks for the face region, then
		# convert the landmark (x, y)-coordinates to a NumPy array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		try:
			# convert dlib's rectangle to a OpenCV-style bounding box
			# [i.e., (x, y, w, h)], then draw the face bounding box
			(x, y, w, h) = face_utils.rect_to_bb(rect)
			cv2.rectangle(resized, (x, y), (x + w, y + h), (0, 255, 0), 2)
			# show the face number
			cv2.putText(resized, "Face #{}".format(i + 1), (x - 10, y - 10),
						cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
			# loop over the (x, y)-coordinates for the facial landmarks
			# and draw them on the image
			for (num, (x, y)) in enumerate(shape):
				cv2.circle(resized, (x, y), 1, (0, 0, 255), -1)
				#cv2.putText(resized, str(num + 1), (x - 10, y - 10),
					#cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 1)

			#cv2.imwrite('images/face_detected.jpg', resized)
			#cv2.imshow("Marked out face", resized)
			#cv2.waitKey(0)


			#Geometrical Features: facial height/ facial width = increases
			## Right Eye & Brow
			r1 = euclidean_dist(shape[0], shape[36]) / euclidean_dist(shape[17], shape[36])
			r2 = euclidean_dist(shape[0], shape[17]) / euclidean_dist(shape[0], shape[36]) #3
			r3 = euclidean_dist(shape[0], shape[36]) / euclidean_dist(shape[1], shape[36])
			r4 = euclidean_dist(shape[0], shape[1]) / euclidean_dist(shape[1], shape[36])
			r5 = euclidean_dist(shape[1], shape[36]) / euclidean_dist(shape[2], shape[36]) #6
			r6 = euclidean_dist(shape[1], shape[2]) / euclidean_dist(shape[2], shape[36])
			r7 = euclidean_dist(shape[17], shape[36]) / euclidean_dist(shape[17], shape[21])
			r8 = euclidean_dist(shape[36], shape[21]) / euclidean_dist(shape[17], shape[36]) #10
			r9 = euclidean_dist(shape[39], shape[17]) / euclidean_dist(shape[39], shape[21])
			r10 = euclidean_dist(shape[39], shape[21]) / euclidean_dist(shape[17], shape[21]) #13
			## Left Eye & Brow
			r11 = euclidean_dist(shape[45], shape[16]) / euclidean_dist(shape[45], shape[26])
			r12 = euclidean_dist(shape[16], shape[26]) / euclidean_dist(shape[26], shape[45])
			r13 = euclidean_dist(shape[45], shape[16]) / euclidean_dist(shape[45], shape[15])
			r14 = euclidean_dist(shape[45], shape[15]) / euclidean_dist(shape[15], shape[16])
			r15 = euclidean_dist(shape[45], shape[15]) / euclidean_dist(shape[45], shape[14])
			r16 = euclidean_dist(shape[45], shape[14]) / euclidean_dist(shape[14], shape[15])
			r17 = euclidean_dist(shape[45], shape[26]) / euclidean_dist(shape[22], shape[45])
			r18 = euclidean_dist(shape[22], shape[26]) / euclidean_dist(shape[22], shape[45])
			r19 = euclidean_dist(shape[42], shape[22]) / euclidean_dist(shape[42], shape[26])
			r20 = euclidean_dist(shape[42], shape[26]) / euclidean_dist(shape[22], shape[26])
			## Eyes, Brows & Nose
			r21 = euclidean_dist(shape[39], shape[21]) / euclidean_dist(shape[39], shape[27])
			r22 = euclidean_dist(shape[39], shape[27]) / euclidean_dist(shape[21], shape[27])
			r23 = euclidean_dist(shape[27], shape[22]) / euclidean_dist(shape[27], shape[42])
			r24 = euclidean_dist(shape[27], shape[42]) / euclidean_dist(shape[42], shape[22])
			r25 = euclidean_dist(shape[39], shape[42]) / euclidean_dist(shape[27], shape[39])
			r26 = euclidean_dist(shape[27], shape[42]) / euclidean_dist(shape[39], shape[42])
			r27 = euclidean_dist(shape[39], shape[42]) / euclidean_dist(shape[39], shape[30])
			r28 = euclidean_dist(shape[39], shape[42]) / euclidean_dist(shape[42], shape[30])
			r29 = euclidean_dist(shape[21], shape[22]) / euclidean_dist(shape[21], shape[27])
			r30 = euclidean_dist(shape[21], shape[22]) / euclidean_dist(shape[22], shape[27])
			## Nose
			r31 = euclidean_dist(shape[27], shape[30]) / euclidean_dist(shape[27], shape[35])
			r32 = euclidean_dist(shape[27], shape[30]) / euclidean_dist(shape[30], shape[35])
			r33 = euclidean_dist(shape[27], shape[30]) / euclidean_dist(shape[27], shape[31])
			r34 = euclidean_dist(shape[27], shape[30]) / euclidean_dist(shape[30], shape[31])
			r35 = euclidean_dist(shape[31], shape[35]) / euclidean_dist(shape[30], shape[35])
			r36 = euclidean_dist(shape[31], shape[35]) / euclidean_dist(shape[30], shape[31])
			r37 = euclidean_dist(shape[31], shape[35]) / euclidean_dist(shape[31], shape[33])
			r38 = euclidean_dist(shape[31], shape[35]) / euclidean_dist(shape[33], shape[35])
			r39 = euclidean_dist(shape[32], shape[34]) / euclidean_dist(shape[27], shape[32])
			r40 = euclidean_dist(shape[32], shape[34]) / euclidean_dist(shape[27], shape[34])
			## Nose & Inner Eye points
			r41 = euclidean_dist(shape[30], shape[31]) / euclidean_dist(shape[39], shape[30])
			r42 = euclidean_dist(shape[30], shape[31]) / euclidean_dist(shape[39], shape[31])
			r43 = euclidean_dist(shape[30], shape[35]) / euclidean_dist(shape[42], shape[30])
			r44 = euclidean_dist(shape[30], shape[35]) / euclidean_dist(shape[42], shape[35])
			## Mouth & Jaw
			r45 = euclidean_dist(shape[48], shape[4]) / euclidean_dist(shape[3], shape[4])
			r46 = euclidean_dist(shape[48], shape[4]) / euclidean_dist(shape[48], shape[3])
			r47 = euclidean_dist(shape[48], shape[4]) / euclidean_dist(shape[48], shape[5])
			r48 = euclidean_dist(shape[4], shape[5]) / euclidean_dist(shape[48], shape[5])
			r49 = euclidean_dist(shape[54], shape[12]) / euclidean_dist(shape[54], shape[13])
			r50 = euclidean_dist(shape[54], shape[12]) / euclidean_dist(shape[12], shape[13])
			r51 = euclidean_dist(shape[54], shape[12]) / euclidean_dist(shape[54], shape[11])
			r52 = euclidean_dist(shape[11], shape[12]) / euclidean_dist(shape[54], shape[11])
			r53 = euclidean_dist(shape[48], shape[51]) / euclidean_dist(shape[48], shape[57])
			r54 = euclidean_dist(shape[48], shape[51]) / euclidean_dist(shape[57], shape[51])
			r55 = euclidean_dist(shape[54], shape[51]) / euclidean_dist(shape[57], shape[51])
			r56 = euclidean_dist(shape[54], shape[51]) / euclidean_dist(shape[54], shape[57])
			r57 = euclidean_dist(shape[48], shape[8]) / euclidean_dist(shape[48], shape[5])
			r58 = euclidean_dist(shape[48], shape[8]) / euclidean_dist(shape[5], shape[8])
			r59 = euclidean_dist(shape[54], shape[8]) / euclidean_dist(shape[54], shape[11])
			r60 = euclidean_dist(shape[54], shape[8]) / euclidean_dist(shape[11], shape[8])
			## Jaw radiuses with landmark [27] as center point
			r61 = euclidean_dist(shape[27], shape[0]) / euclidean_dist(shape[0], shape[1])
			r62 = euclidean_dist(shape[27], shape[0]) / euclidean_dist(shape[27], shape[1])
			r63 = euclidean_dist(shape[27], shape[1]) / euclidean_dist(shape[1], shape[2])
			r64 = euclidean_dist(shape[27], shape[1]) / euclidean_dist(shape[27], shape[2])
			r65 = euclidean_dist(shape[27], shape[2]) / euclidean_dist(shape[2], shape[3])
			r66 = euclidean_dist(shape[27], shape[2]) / euclidean_dist(shape[27], shape[3])
			r67 = euclidean_dist(shape[27], shape[3]) / euclidean_dist(shape[3], shape[4])
			r68 = euclidean_dist(shape[27], shape[3]) / euclidean_dist(shape[27], shape[4])
			r69 = euclidean_dist(shape[27], shape[4]) / euclidean_dist(shape[4], shape[5])
			r70 = euclidean_dist(shape[27], shape[4]) / euclidean_dist(shape[27], shape[5])
			r71 = euclidean_dist(shape[27], shape[5]) / euclidean_dist(shape[5], shape[6])
			r72 = euclidean_dist(shape[27], shape[5]) / euclidean_dist(shape[27], shape[6])
			r73 = euclidean_dist(shape[27], shape[6]) / euclidean_dist(shape[6], shape[7])
			r74 = euclidean_dist(shape[27], shape[6]) / euclidean_dist(shape[27], shape[7])
			r75 = euclidean_dist(shape[27], shape[7]) / euclidean_dist(shape[7], shape[8])
			r76 = euclidean_dist(shape[27], shape[7]) / euclidean_dist(shape[27], shape[8])
			r77 = euclidean_dist(shape[27], shape[8]) / euclidean_dist(shape[8], shape[9])
			r78 = euclidean_dist(shape[27], shape[8]) / euclidean_dist(shape[27], shape[9])
			r79 = euclidean_dist(shape[27], shape[9]) / euclidean_dist(shape[9], shape[10])
			r80 = euclidean_dist(shape[27], shape[9]) / euclidean_dist(shape[27], shape[10])
			r81 = euclidean_dist(shape[27], shape[10]) / euclidean_dist(shape[10], shape[11])
			r82 = euclidean_dist(shape[27], shape[10]) / euclidean_dist(shape[27], shape[11])
			r83 = euclidean_dist(shape[27], shape[11]) / euclidean_dist(shape[11], shape[12])
			r84 = euclidean_dist(shape[27], shape[11]) / euclidean_dist(shape[27], shape[12])
			r85 = euclidean_dist(shape[27], shape[12]) / euclidean_dist(shape[12], shape[13])
			r86 = euclidean_dist(shape[27], shape[12]) / euclidean_dist(shape[27], shape[13])
			r87 = euclidean_dist(shape[27], shape[13]) / euclidean_dist(shape[13], shape[14])
			r88 = euclidean_dist(shape[27], shape[13]) / euclidean_dist(shape[27], shape[14])
			r89 = euclidean_dist(shape[27], shape[14]) / euclidean_dist(shape[14], shape[15])
			r90 = euclidean_dist(shape[27], shape[14]) / euclidean_dist(shape[27], shape[15])
			r91 = euclidean_dist(shape[27], shape[15]) / euclidean_dist(shape[15], shape[16])
			r92 = euclidean_dist(shape[27], shape[15]) / euclidean_dist(shape[27], shape[16])
			r93 = euclidean_dist(shape[27], shape[16]) / euclidean_dist(shape[15], shape[16])

			# Extracted features (Hog features + Geometrical features)
			parts = {'1': r1, '2': r2, '3': r3, '4': r4, '5': r5, '6': r6, '7': r7, '8': r8, '9': r9, '10': r10,
					 '11': r11, '12': r12, '13': r13, '14': r14, '15': r15, '16': r16, '17': r17, '18': r18, '19': r19, '20': r20,
					 '21': r21, '22': r22, '23': r23, '24': r24, '25': r25, '26': r26, '27': r27, '28': r28, '29': r29, '30': r30,
					 '31': r31, '32': r32, '33': r33, '34': r34, '35': r35, '36': r36, '37': r37, '38': r38, '39': r39, '40': r40,
					 '41': r41, '42': r42, '43': r43, '44': r44, '45': r45, '46': r46, '47': r47, '48': r48, '49': r49, '50': r50,
					 '51': r51, '52': r52, '53': r53, '54': r54, '55': r55, '56': r56, '57': r57, '58': r58, '59': r59, '60': r60,
					 '61': r61, '62': r62, '63': r63, '64': r64, '65': r65, '66': r66, '67': r67, '68': r68, '69': r69, '70': r70,
					 '71': r71, '72': r72, '73': r73, '74': r74, '75': r75, '76': r76, '77': r77, '78': r78, '79': r79, '80': r80,
					 '81': r81, '82': r82, '83': r83, '84': r84, '85': r85, '86': r86, '87': r87, '88': r88, '89': r89, '90': r90,
					 '91': r91, '92': r92, '93': r93}

			face_dict = parts.copy()

			# Getting the name the image is saved with
			filename = os.path.basename(file)
			# Getting part of the name which is the age of the individual in the image
			age = filename.split('.')[0].split('A')[1].split('a')[0].split('b')[0]
			if int(age) >= 12:
				face_dict['label'] = 'Adult'
			else:
				face_dict['label'] = 'Child'

			data.append(face_dict.copy())

		except Exception as e:
			print(str(e))

with open('93ratios.pickle', 'wb') as f:
  pickle.dump(data, f)




