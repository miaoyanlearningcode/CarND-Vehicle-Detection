import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pickle
from sklearn.model_selection import train_test_split
from skimage.feature import hog

def dataRead(pklPath):
	with open(pklPath, 'rb') as pfile:
		data = pickle.load(pfile)

	veh   = data['vehicles']
	noVeh = data['nonVeh']
	del data

	carImg = []
	notCarImg = []
	
	for imgPath in veh:
		carImg.append(mpimg.imread(imgPath))

	for imgPath in noVeh:
		notCarImg.append(mpimg.imread(imgPath))
	
	carImg = np.array(carImg)
	notCarImg = np.array(notCarImg)
	return carImg, notCarImg

def color_hist(img, nbins=32, bins_range=(0, 256)):
	# Compute the histogram of the RGB channels separately
	ch1Hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
	ch2Hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
	ch3Hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((ch1Hist[0], ch2Hist[0], ch3Hist[0]))
	# Return the individual histograms, bin_centers and feature vector
	return hist_features
 
def bin_spatial(img, size=(32, 32)):            
	# Use cv2.resize().ravel() to create the feature vector
	features = cv2.resize(img, size).ravel() 
	# Return the feature vector
	return features   

def get_hog_features(img, orient, pix_per_cell, cell_per_block,vis=False, feature_vec=True):
	# Call with two outputs if vis==True
	if vis == True:
		features, hog_image = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
									cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True, 
									visualise=vis, feature_vector=feature_vec)
		return features, hog_image
		# Otherwise call with one output
	else:      
		features = hog(img, orientations=orient, pixels_per_cell=(pix_per_cell, pix_per_cell),
						cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
						visualise=vis, feature_vector=feature_vec)
		return features

def single_img_features(img, cspace='RGB', spatial_size=(32, 32),
						hist_bins=32, orient=9, 
						pix_per_cell=8, cell_per_block=2, hog_channel=0,
						spatial_feat=True, hist_feat=True, hog_feat=True):  
	# Create a list to append feature vectors to
	features = []

	if cspace != 'RGB':
			if cspace == 'HSV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
			elif cspace == 'LUV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
			elif cspace == 'HLS':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
			elif cspace == 'YUV':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
			elif cspace == 'YCrCb':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
			elif cspace == 'GRAY':
				feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	else: feature_image = np.copy(img)

	# histogram feature
	if hist_feat == True:
		hist_features = color_hist(feature_image, nbins=hist_bins)
		features.append(hist_features)

	if spatial_feat == True:
		spat_features = bin_spatial(feature_image, size=spatial_size)
		features.append(spat_features)

	if hog_feat == True: 
		# Call get_hog_features() with vis=False, feature_vec=True
		if cspace == 'GRAY':
			hog_features=get_hog_features(feature_image, orient, pix_per_cell, 
				cell_per_block, vis=False, feature_vec=True)
		elif hog_channel == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.append(get_hog_features(feature_image[:,:,channel], 
								orient, pix_per_cell, cell_per_block, 
								vis=False, feature_vec=True))
			hog_features = np.ravel(hog_features)        
		else:
			hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
					pix_per_cell, cell_per_block, vis=False, feature_vec=True)
				# Append the new feature vector to the features list
		features.append(hog_features)
	# Return list of feature vectors
	return np.concatenate(features)
  


def extract_features(imgs, cspace='RGB', bin_size = (32, 32), hist_bins = 32, orient=9, 
						pix_per_cell=8, cell_per_block=2, hog_channel=0, binFeat = True, histFeat = True, hogFeat = True):
	# Create a list to append feature vectors to
	features = []
	# Iterate through the list of images
	for img in imgs:
		featureTmp = []

		if cspace != 'RGB':
				if cspace == 'HSV':
					feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
				elif cspace == 'LUV':
					feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
				elif cspace == 'HLS':
					feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
				elif cspace == 'YUV':
					feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
				elif cspace == 'YCrCb':
					feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
				elif cspace == 'GRAY':
					feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
		else: feature_image = np.copy(img)

		# histogram feature
		if histFeat == True:
			hist_features = color_hist(feature_image, nbins=hist_bins)
			featureTmp.append(hist_features)

		if binFeat == True:
			spat_features = bin_spatial(feature_image, size=bin_size)
			featureTmp.append(spat_features)

		if hogFeat == True: 
			# Call get_hog_features() with vis=False, feature_vec=True
			if cspace == 'GRAY':
				hog_features = get_hog_features(feature_image, orient, 
					pix_per_cell, cell_per_block, vis=False, feature_vec=True)
			elif hog_channel == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:,:,channel], 
									orient, pix_per_cell, cell_per_block, 
									vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)        
			else:
				hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
					pix_per_cell, cell_per_block, vis=False, feature_vec=True)
			# Append the new feature vector to the features list
			featureTmp.append(hog_features)
		features.append(np.concatenate(featureTmp))
	# Return list of feature vectors
	return features

# Define a function that takes an image,
# start and stop positions in both x and y, 
# window size (x and y dimensions),  
# and overlap fraction (for both x and y)
def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None], 
					xy_window=(64, 64), xy_overlap=(0.5, 0.5)):
	# If x and/or y start/stop positions not defined, set to image size
	if x_start_stop[0] == None:
		x_start_stop[0] = 0
	if x_start_stop[1] == None:
		x_start_stop[1] = img.shape[1]
	if y_start_stop[0] == None:
		y_start_stop[0] = 0
	if y_start_stop[1] == None:
		y_start_stop[1] = img.shape[0]
	
	# Compute the span of the region to be searched    
	xspan = x_start_stop[1] - x_start_stop[0]
	yspan = y_start_stop[1] - y_start_stop[0]
	# Compute the number of pixels per step in x/y
	nx_pix_per_step = np.int(xy_window[0]*(1 - xy_overlap[0]))
	ny_pix_per_step = np.int(xy_window[1]*(1 - xy_overlap[1]))
	# Compute the number of windows in x/y
	nx_buffer = np.int(xy_window[0]*(xy_overlap[0]))
	ny_buffer = np.int(xy_window[1]*(xy_overlap[1]))
	nx_windows = np.int((xspan-nx_buffer)/nx_pix_per_step) 
	ny_windows = np.int((yspan-ny_buffer)/ny_pix_per_step) 
	# Initialize a list to append window positions to
	window_list = []
	# Loop through finding x and y window positions
	# Note: you could vectorize this step, but in practice
	# you'll be considering windows one by one with your
	# classifier, so looping makes sense
	for ys in range(ny_windows):
		for xs in range(nx_windows):
		# Calculate window position
			startx = xs*nx_pix_per_step + x_start_stop[0]
			endx = startx + xy_window[0]
			starty = ys*ny_pix_per_step + y_start_stop[0]
			endy = starty + xy_window[1]
			# Append window position to list
			window_list.append(((startx, starty), (endx, endy)))
	# Return the list of windows
	return window_list

def scaledWindows(img, xy_overlap=(0.7,0.7)):
	x_ss = [None, None]
	y_ss = [
		[400, 656],
		[400, 656],
		[400, 560],
		[400, 528]
	]
	window_xy = [(128, 128),(96, 96),(80, 80),(64, 64)]
	window_list = []
	for i in range(4):
		xy = window_xy[i]
		windows = slide_window(img, x_start_stop=x_ss, y_start_stop=y_ss[i], xy_window=xy, xy_overlap = xy_overlap)
		window_list.extend(windows)

	return window_list

def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
	# Make a copy of the image
	imcopy = np.copy(img)
	# Iterate through the bounding boxes
	for bbox in bboxes:
		# Draw a rectangle given bbox coordinates
		cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
	# Return the image copy with boxes drawn
	return imcopy


# Define a function you will pass an image 
# and the list of windows to be searched (output of slide_windows())
def search_windows(img, windows, clf, scaler, color_space='RGB', 
					spatial_size=(32, 32), hist_bins=32, 
					hist_range=(0, 256), orient=9, 
					pix_per_cell=8, cell_per_block=2, 
					hog_channel=0, spatial_feat=True, 
					hist_feat=True, hog_feat=True):

	#1) Create an empty list to receive positive detection windows
	on_windows = []
	#2) Iterate over all windows in the list
	for window in windows:
		#3) Extract the test window from original image
		test_img = cv2.resize(img[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))      
		#4) Extract features for that window using single_img_features()
		features = single_img_features(test_img, cspace=color_space, 
			spatial_size=spatial_size, hist_bins=hist_bins, 
			orient=orient, pix_per_cell=pix_per_cell, 
			cell_per_block=cell_per_block, 
			hog_channel=hog_channel, spatial_feat=spatial_feat, 
			hist_feat=hist_feat, hog_feat=hog_feat)
		#5) Scale extracted features to be fed to classifier
		test_features = scaler.transform(np.array(features).reshape(1, -1))
		#6) Predict using your classifier
		prediction = clf.predict(test_features)
		#7) If positive (prediction == 1) then save the window
		if prediction == 1:
			on_windows.append(window)
	#8) Return windows for positive detections
	return on_windows


def add_heat(heatmap, bbox_list):
	# Iterate through list of bboxes
	for box in bbox_list:
	# Add += 1 for all pixels inside each bbox
	# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	# Return updated heatmap
	return heatmap# Iterate through list of bboxes

def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap

def draw_labeled_bboxes(img, labels):
	# Iterate through all detected cars
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		# Draw the box on the image
		cv2.rectangle(img, bbox[0], bbox[1], (0,0,1), 6)
	# Return the image
	return img