## Vehicle Detection Project

---

**Vehicle Detection Project**

The goals of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run your pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[image1]: ./output_images/car_non_car.png
[image2]: ./output_images/car_features_RGB.png
[image3]: ./output_images/car_features_HLS.png
[image4]: ./output_images/car_features_HSV.png
[image5]: ./output_images/car_features_YCrCb.png
[image6]: ./output_images/car_features_YUV.png
[image7]: ./output_images/noncar_features_RGB.png
[image8]: ./output_images/noncar_features_HLS.png
[image9]: ./output_images/noncar_features_HSV.png
[image10]: ./output_images/noncar_features_YCrCb.png
[image11]: ./output_images/noncar_features_YUV.png
[image12]: ./output_images/detection_1.png
[image13]: ./output_images/detection_2.png
[image14]: ./output_images/detection_3.png
[image15]: ./output_images/detection_4.png
[image16]: ./output_images/detection_5.png
[image17]: ./output_images/detection_6.png
[image18]: ./output_images/heatmap_1.png
[image19]: ./output_images/heatmap_2.png
[image20]: ./output_images/heatmap_3.png
[image21]: ./output_images/heatmap_4.png
[image22]: ./output_images/heatmap_5.png
[image23]: ./output_images/heatmap_6.png
[image24]: ./output_images/processed_1.png
[image25]: ./output_images/processed_2.png
[image26]: ./output_images/processed_3.png
[image27]: ./output_images/processed_4.png
[image28]: ./output_images/processed_5.png
[image29]: ./output_images/processed_6.png


---
## Overview of approach

The objective of this project was to apply an SVM learning technique on a video stream to detect and locate cars. To train the SVM, a training and testing set of car and non-car images was provided by Udacity. An sample from the training data set is provided:

![alt text][image1]

The SVM was trained using a combination of Histogram of Oriented Gradients (HOG) features, spatially binned features, and histogram of color features.

All the code is contained in the file `Project.py`. The file contains fully defined functions for each aspect of the project.

The final implementation of the feature extractor consisted of the following features:
- HOG features
- Color space: YUV
- Hog Orientations: 9
- HOG pixels per cell: 8
- HOG cells per block: 2
- HOG channels: All
- Spatial binning dimensions: (32, 32)
- Number of histogram bins: 32  

The sliding window technique was replaced with a HOG sub-sampling window approach recommended in the lecture notes.

### SVM Training

#### HOG
The following figures represent different responses of the HOG features on image channels for car, and non-car images for comparison. The selection of optimal parameters is described in the next section. HOG features were extracted using the `get_hog_features()` function on line 27.

Spatial features were recovered using the `bin_spatial()` function on line 47. Color histogram features were extracted using `color_hist()` function on line 63.

Feature extraction functions were defined for training over a list of images on line 74: `extract_features()`. Features on individual images were extracted with `single_img_features()` on line 131.
Both feature extraction functions allow for selecting color spaces and feature parameters.


![alt text][image2]

Car RGB HOG Features

![alt text][image3]

Car HLS HOG Features

![alt text][image4]

Car HSV HOG Features

![alt text][image5]

Car YCrCb HOG Features

![alt text][image6]

Car YUV HOG Features

![alt text][image7]

Non-car RGB HOG Features

![alt text][image8]

Non-car HLS HOG Features

![alt text][image9]

Non-car HSV HOG Features

![alt text][image10]

Non-car YCrCb HOG Features

![alt text][image11]

Non-car YUV HOG Features


#### Optimal Parameter Selection & Training
The optimal set of parameters for the features was chosen by recurrently testing various parameters until the best test result was achieved in training. Various combinations of features on different color channels were tested for SVM training. The selected parameters (described above) were chosen after multiple training runs were conducted.

The function `trainSVC()` was used with manually selected parameters until the desired testing accuracy was achieved. The initial parameter section used a subset of the available data for training and testing. Once the ideal parameters were selected, the complete dataset was used for training. The selected parameters consistently returned testing accuracy over 95%. The trained SVC was saved to allow quick recall in subsequent functions using `loadTrainedSVC()` on line 263.

### Implementation of Vehicle Detection
Once the SVC was trained, a detection framework was developed. The framework consisted of three main steps:
1. Find all detected vehicles in an image and return detection boxes.
2. Filter outliers
3. Plot results.

The complete framework was wrapped into a `detector()` class defined on line 408. Other parts of the `detector()` include loading the training data.

#### 1. Vehicle Detection
The vehicle detection implemented using a HOG sub-sampling window approach described in the lecture notes. The sub-sampling partitioned a part of the image into cells, and applied the HOG transform to each cell, and applied the spatial and color histogram feature extractor to larger patches. The input image was cropped in the y-axis to reduce the search space above the front hood, and below the image horizon. Each image was cropped on the y-axis between (400,654) based on the general location of the hood and horizon. The sub-sampling search scale was set to 1.5 based on manual testing for best detections.

The detection code was defined in the `find_cars()` function at line 315. This function applied the detector to a region of interest in the image, and returned all rectangles corresponding to image detection:


![alt text][image12]

Test image 1</p>

![alt text][image13]

Test image 2

![alt text][image14]

Test image 3

![alt text][image15]

Test image 4

![alt text][image16]

Test image 5

![alt text][image17]

Test image 6

#### 2. Filter outliers
The multiple detections produced by `find_Cars()` must be filtered to remove false positives and overlapping detections. This filtering was done by the `filterBoxList()` function on line 383. This function created a heat map image of the hit-boxes for the cars. Then, each boxed region was used to increase the pixel-value on the heat map image. A thresholding was applied to remove false-positives with a user-defined threshold value. This produced the following images:


![alt text][image18]

Test image 1 heatmap

![alt text][image19]

Test image 2 heatmap

![alt text][image20]

Test image 3 heatmap

![alt text][image21]

Test image 4 heatmap

![alt text][image22]

Test image 5 heatmap

![alt text][image23]

Test image 6 heatmap

#### 3. Plot results
The complete pipeline resulted in the following output detections:

![alt text][image24]

Test image 1 processed

![alt text][image25]

Test image 2 processed

![alt text][image26]

Test image 3 processed

![alt text][image27]

Test image 4 processed

![alt text][image28]

Test image 5 processed

![alt text][image29]

Test image 6 processed

#### Video
The processed video of the implemented method can be found [here]('output_images/project_video.mp4').



### Discussion
The methodology developed worked fairly well for the project video. Several issues are herein highlighted for future consideration.

#### 1. False Positives
False positives were still detected even when using the heat map. This specifically occurs when vehicles are detected in incoming lanes, or in dark regions of the image. Dark regions can be countered using contrast equalization, or simply rejection of detection in image regions with low light. Oncoming vehicles correctly detected are technically true positives, and may are useful for self-driving cars.

#### 2. Detection Size
The size of detections in certain parts of the video are much smaller than the actual vehicle. This was due to the heat map threshold setting. This could be improved with further threshold parameter turning, or individual tracking of detected vehicles.

#### 3. Overlapping Detections
Overlapping detections occurred when the heat map threshold incorrectly separated two close-by vehicles. To counter this issue, several possible solutions could be implemented:
1. robust tracking of individual vehicles such as Kalman filtering or particle filter in the image plane.
2. improve segmentation through local tracking techniques such as optical flow.

#### Future Development Recommendation
The following ideas could be implemented to improve the vehicle detection:
1. create individual instances of detected car objects with tracking capabilities to improve tracking smoothness and avoid merging detections for close vehicles.
2. add local tracking techniques around detected vehicles (such as optical flow) to improve boundary separation between the vehicles and the background.
3. add a Kalman Filter or a particle filter for each detected vehicle.
