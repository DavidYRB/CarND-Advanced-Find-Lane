import numpy as np
import cv2

class tracker(object):
	"""docstring for tracker"""
	def __init__(self, window_width, window_height, margin, ym2pixel=1, xm2pixel=1, smooth_factor=15):

		self.centroids = []
		self.window_height = window_height
		self.window_width = window_width
		self.margin = margin
		self.ym_per_pixel = ym2pixel
		self.xm_per_pixel = xm2pixel
		self.smooth_factor = smooth_factor

	def find_window_centroids(self, img):
		window_width = self.window_width
		window_height = self.window_height
		margin = self.margin
		offset = window_width/2
		window_centroids = []
		window = np.ones(window_width)

		# find the first pair of centroids
	    # since the max value of convolution happens on the right side of the window, so the center index of the window
        # should be the index of the max convolution subtract have the the window 
		l_sum = np.sum(img[int(3*img.shape[0]/4):,:int(img.shape[1]/2)], axis=0)
		l_center = np.argmax(np.convolve(window, l_sum)) - offset
		r_sum = np.sum(img[int(3*img.shape[0]/4):,int(img.shape[1]/2):], axis=0)
		r_center = np.argmax(np.convolve(window, r_sum )) + int(img.shape[1]/2) - offset
		
		# append first pair centroids
		window_centroids.append((l_center, r_center))
        

		for level in range(1, int(img.shape[0]/window_height)):
			image_layer = np.sum(img[img.shape[0]-(level+1)*window_height:img.shape[0]-level*window_height, :], axis=0)
			conv_sig = np.convolve(window, image_layer)

			# find left controid in current layer
			l_min_index = int(max(l_center+offset-margin, 0))
			l_max_index = int(min(l_center+offset+margin, img.shape[1]))
			l_center = np.argmax(conv_sig[l_min_index:l_max_index]) + l_min_index - offset

			# find right controid in current layer
			r_min_index = int(max(r_center+offset-margin, 0))
			r_max_index = int(min(r_center+offset+margin, img.shape[1]))
			r_center = np.argmax(conv_sig[r_min_index:r_max_index]) + r_min_index - offset

			# append centroids of current layer
			window_centroids.append((l_center, r_center))

		self.centroids.append(window_centroids)
		return np.average(self.centroids[-self.smooth_factor:], axis=0)

