import cv2
import pickle
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import glob

# import image
image_names = glob.glob('./camera_cal/*.jpg')

def find_corners(image_names):
	objpoints = []
	imgpoints = []
	
	objp = np.zeros((6*9, 3), np.float32)
	objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

	for idx, name in enumerate(image_names):
		img = mpimg.imread(name)
		gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		# find corners
		ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)

		if ret == True:
			print("Working on " + name)
			cv2.drawChessboardCorners(img, (9,6), corners, ret)
			new_name = "find_corner"+ str(idx) +".jpg"
			cv2.imwrite("./cornered_camera_cal/" + new_name, img)
			objpoints.append(objp)
			imgpoints.append(corners)

	return objpoints, imgpoints

# find, draw corners and save to new images
objpoints, imgpoints = find_corners(image_names)

# use a calibration image to calcualte undistort matrices
img = cv2.imread("./camera_cal/calibration16.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

# save matrices to a pickled file.
undis_pickle = {'mtx': mtx, 'dist': dist}
pickle.dump(undis_pickle, open('undis_pickle.p', 'wb'))




