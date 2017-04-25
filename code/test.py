import dataProcess as dp
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from sklearn.externals import joblib

# car, notCar = dp.dataSplit('../data/allData.p')

# print(car.shape)
# print(notCar.shape)

img = mpimg.imread('../test_images/test1.jpg')
# plt.imshow(img)
svc = joblib.load('../model/svmCar.pkl')
windowsList = dp.scaledWindows(img)
img = dp.draw_boxes(img, windowsList)