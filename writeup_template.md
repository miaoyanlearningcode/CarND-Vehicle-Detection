
**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/carNocar.png
[image2]: ./output_images/carNocarHOG.png
[image3]: ./output_images/slidingWindows.png
[image4]: ./output_images/test1Init.png
[image5]: ./output_images/test2Init.png
[image6]: ./output_images/test3Init.png
[image7]: ./output_images/test4Init.png
[image8]: ./output_images/test5Init.png
[image9]: ./output_images/test6Init.png
[image10]: ./output_images/heat5.png
[image11]: ./output_images/heat6.png
[image12]: ./output_images/label5.png
[image13]: ./output_images/label6.png
[image14]: ./output_images/test5.png
[image15]: ./output_images/test6.png
[video1]: ./project_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
###Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
###Writeup / README

####1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

###Histogram of Oriented Gradients (HOG)

####1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(1, 1)`:


![alt text][image2]

####2. Explain how you settled on your final choice of HOG parameters.

I tried various combinations of parameters and the final parameters are `orientations=9`, `pixels_per_cell=(16, 16)` and `cells_per_block=(1, 1)`

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

The training process is in `training.py`. THe first step is to extract the features of HOG, color histogram and spatial binning and the labels. Using `StandardScaler` to normalized the features and split the data into training and testing data. Apply `LinearSVC` to training data and get a model `svc`. Test the accuracy of pridictions for testing data. 

Save the model, configration and scaling information into pickle file so that we could reuse it. 

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The sliding windows can be seen in `cell3` and the functions used for it are defined in `dataProcess.py`. The functions are `slide_window` and  `scaledWindow`. We only focus on the lower half image. Based on the size of window, choose the specific staring and stop values in y axis. For example, here I choose 4 sizes of windows, 128,96,80 and 64 and the `y_start_stop` I choose are `y_ss = [[400, 656],[400, 656],[400, 560],[400, 528]]`. And the overlap rate is 0.7. 

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

The pipeline is as following:
1. convert color image from RGB to YCrCb
2. create a list of boxes to search cars
3. for each box, generate the features we need. The features are same as the features in training process. 3-channel HOG features plus spatially binned color and histograms of color in the feature vector.
4. apply the svm model to predict and keep the boxes with the prediction of 1. Here are some example images without adding tracking:

![alt text][image4]
![alt text][image5]
![alt text][image6]
![alt text][image7]
![alt text][image8]
![alt text][image9]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_output.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are 5th and 6th test images' corresponding heatmaps:

![alt text][image10]
![alt text][image11]

### Here are the outputs of `scipy.ndimage.measurements.label()` on the integrated heatmap from 5th and 6th test images:
![alt text][image12]
![alt text][image13]

### Here the resulting bounding boxes are drawn onto the 5th and 6th test images in the series:
![alt text][image14]
![alt text][image15]

---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

In the 5th test image, we can see that the white car is not detectable. So the training can be better if we have more data or if we can try CNN.

The tracking algorithm can be improved. For example, we could apply Kalman Filter to track the car. Or maybe we can combine the detection and tracking together. Another thing is that we could save the cars that we detected into dataset for later use. 

