import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import pickle
from sklearn.model_selection import train_test_split
import dataProcess as dp
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
cspace = 'YCrCb'
spatial_size = (16,16)
hist_bins = 16
orient = 9
pix_per_cell = 8
cell_per_block = 1
hog_channel = 'ALL'
spatial_feat = True
hist_feat = True
hog_feat = True

cars, notCars = dp.dataRead('../data/allData.p')

carFeature = dp.extract_features(cars, cspace =cspace, bin_size = spatial_size, hist_bins = hist_bins, 
						orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
						hog_channel=hog_channel,binFeat = spatial_feat, histFeat = hist_feat, 
						hogFeat = hog_feat)
# print(carFeature)
notCarFeature = dp.extract_features(notCars,  cspace =cspace, bin_size = spatial_size, hist_bins = hist_bins, 
						orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
						hog_channel=hog_channel,binFeat = spatial_feat, histFeat = hist_feat, 
						hogFeat = hog_feat)
# print(notCarFeature)
X = np.vstack((carFeature, notCarFeature)).astype(np.float64)                        
X_scaler = StandardScaler().fit(X)
scaled_X = X_scaler.transform(X)
y = np.hstack((np.ones(len(carFeature)), np.zeros(len(notCarFeature))))

rand_state = np.random.randint(0, 100)
X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.2, random_state=rand_state)


parameter = dict(color_space = cspace,spatial_size=spatial_size, hist_bins=hist_bins, 
	orient=orient, pix_per_cell=pix_per_cell, cell_per_block=cell_per_block, 
	hog_channel=hog_channel, spatial_feat=spatial_feat, 
	hist_feat=hist_feat, hog_feat=hog_feat)

svc = LinearSVC()
svc.fit(X_train, y_train)
print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))

joblib.dump({'model':svc, 'config':parameter, 'scaling':X_scaler}, '../model/svmCar.pkl')
print ('save model success')