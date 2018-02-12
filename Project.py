#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 16:49:41 2018

@author: en
"""
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import glob
import imageio
import pickle
import os.path
from moviepy.editor import VideoFileClip
#from IPython.display import HTML
#from IPython.display import clear_output
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from sklearn.cross_validation import train_test_split
from scipy.ndimage.measurements import label

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel() 
    # Return the feature vector
    return features

# color convertion function
def convert_color(img, conv='RGB2YCrCb'):
    if conv == 'RGB2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    if conv == 'BGR2YCrCb':
        return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
    if conv == 'RGB2LUV':
        return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)

# Compute color histograms on png images with range (0,1)
def color_hist(img, nbins=32, bins_range=(0, 1)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

# function to extract features from a list of images
def extract_features(imgs, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):
    # Create a list to append feature vectors to
    features = []
    # Iterate through the list of images
    for file in imgs:
        file_features = []
        # Read in each one by one
        image = mpimg.imread(file)
        # apply color conversion if other than 'RGB'
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(image)      
        
        
   
        if spatial_feat == True:
            spatial_features = bin_spatial(feature_image, size=spatial_size)
            file_features.append(spatial_features)
        if hist_feat == True:
            # Apply color_hist()
            hist_features = color_hist(feature_image, nbins=hist_bins)
            file_features.append(hist_features)
        if hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
            if hog_channel == 'ALL':
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
            file_features.append(hog_features)
            temp = np.concatenate(file_features)
        # CHECK IF ANY IN TEMP ARE NAN! Reject!
        if not np.any(np.isnan(temp)):
            features.append(temp)
    # Return list of feature vectors
    return features
    

# Extract features from a single image:
def single_img_features(img, color_space='RGB', spatial_size=(32, 32),
                        hist_bins=32, orient=9, 
                        pix_per_cell=8, cell_per_block=2, hog_channel=0,
                        spatial_feat=True, hist_feat=True, hog_feat=True):    
    #1) Define an empty list to receive features
    img_features = []
    #2) Apply color conversion if other than 'RGB'
    if color_space != 'RGB':
        if color_space == 'HSV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feature_image = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
    else: feature_image = np.copy(img)      
    #3) Compute spatial features if flag is set
    if spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=spatial_size)
        #4) Append features to list
        img_features.append(spatial_features)
    #5) Compute histogram features if flag is set
    if hist_feat == True:
        hist_features = color_hist(feature_image, nbins=hist_bins)
        #6) Append features to list
        img_features.append(hist_features)
    #7) Compute HOG features if flag is set
    if hog_feat == True:
        if hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.extend(get_hog_features(feature_image[:,:,channel], 
                                    orient, pix_per_cell, cell_per_block, 
                                    vis=False, feature_vec=True))      
        else:
            hog_features = get_hog_features(feature_image[:,:,hog_channel], orient, 
                        pix_per_cell, cell_per_block, vis=False, feature_vec=True)
        #8) Append features to list
        img_features.append(hog_features)

    #9) Return concatenated array of features
    return np.concatenate(img_features)

# function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy
    
# Training function for SVC

def trainSVC(vehicles = None, non_vehicles = None, sample_size = -1, save_data = True):
    if vehicles is  None or non_vehicles is None:
        print("Input Error! vehicles or non_vehicles are not defined!\n")
    
    if sample_size>len(vehicles) or sample_size ==-1:
        cars = vehicles
    else:
        cars = vehicles[0:sample_size]
    if sample_size>len(non_vehicles) or sample_size ==-1:
        notcars = non_vehicles
    else:
        notcars = non_vehicles[0:sample_size]   
    color_space = 'YUV' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient = 9  # HOG orientations
    pix_per_cell = 8 # HOG pixels per cell
    cell_per_block = 2 # HOG cells per block
    hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
    spatial_size = (32, 32) # Spatial binning dimensions
    hist_bins = 32    # Number of histogram bins
    spatial_feat = True # Spatial features on or off
    hist_feat = True # Histogram features on or off
    hog_feat = True # HOG features on or off
    y_start_stop = [None, None] # Min and max in y to search in slide_window()
    
    car_features = extract_features(cars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    notcar_features = extract_features(notcars, color_space=color_space, 
                            spatial_size=spatial_size, hist_bins=hist_bins, 
                            orient=orient, pix_per_cell=pix_per_cell, 
                            cell_per_block=cell_per_block, 
                            hog_channel=hog_channel, spatial_feat=spatial_feat, 
                            hist_feat=hist_feat, hog_feat=hog_feat)
    # HERE Remove features with NaN's!
    
    X = np.vstack((car_features, notcar_features)).astype(np.float64)                        
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X)
    # Apply the scaler to X
    scaled_X = X_scaler.transform(X)
    
    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    
    
    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)
    X_train, X_test, y_train, y_test = train_test_split(
        scaled_X, y, test_size=0.2, random_state=rand_state)
    
    print('Using:',orient,'orientations',pix_per_cell,
        'pixels per cell and', cell_per_block,'cells per block')
    print('Feature vector length:', len(X_train[0]))
    # Use a linear SVC 
    svc = LinearSVC()
    # Check the training time for the SVC
    t=time.time()
    svc.fit(X_train, y_train)
    t2 = time.time()
    print(round(t2-t, 2), 'Seconds to train SVC...')
    # Check the score of the SVC
    print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
    # Check the prediction time for a single sample
    t=time.time()
    train_data = {"svc": svc, "X_scaler": X_scaler, "orient":orient,
                   "pix_per_cell": pix_per_cell, "cell_per_block":cell_per_block,
                   "spatial_size":spatial_size, "hist_bins":hist_bins}
    pickle.dump( train_data, open("train_data.p", "wb" ) )
    return svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,hist_bins

def loadTrainedSVC(fileName):
    try:
        # LOAD DATA:
        dist_pickle = pickle.load( open(fileName, "rb" ) )
        svc = dist_pickle["svc"]
        X_scaler = dist_pickle["X_scaler"]
        orient = dist_pickle["orient"]
        pix_per_cell = dist_pickle["pix_per_cell"]
        cell_per_block = dist_pickle["cell_per_block"]
        spatial_size = dist_pickle["spatial_size"]
        hist_bins = dist_pickle["hist_bins"]
        return svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,hist_bins
    except:
        print("Could not load data!\n")

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

# box must come in as np.array(2,2) [x,y],[x,y]
def draw_box(img,box):
    cv2.rectangle(img, (box[0,0],box[0,1]), (box[1,0],box[1,1]), (0,0,255), 6)
    return img


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
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins):
    
    # convert back to 0-1 range
    img = img.astype(np.float32)/255
    
    img_tosearch = img[ystart:ystop,:,:]
    ctrans_tosearch = convert_color(img_tosearch, conv='RGB2YCrCb')
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
        
    ch1 = ctrans_tosearch[:,:,0]
    ch2 = ctrans_tosearch[:,:,1]
    ch3 = ctrans_tosearch[:,:,2]

    # Define blocks and steps as above
    nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
    nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1 
    nfeat_per_block = orient*cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    # Compute individual channel HOG features for the entire image
    hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
    hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
    
    # create list of boxes:
    box_list = []
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            # Extract HOG for this patch
            hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
            hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

            xleft = xpos*pix_per_cell
            ytop = ypos*pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            spatial_features = bin_spatial(subimg, size=spatial_size)
            hist_features = color_hist(subimg, nbins=hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))       
            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                box_list.append(np.array([[xbox_left, ytop_draw+ystart],[xbox_left+win_draw,ytop_draw+win_draw+ystart]]))
                
    return box_list
 

def filterBoxList(img, box_list, heat_thresh = 0.4):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    # Add heat to each box in box list
    heat = add_heat(heat,box_list)
    # Apply threshold to help remove false positives
    heat = apply_threshold(heat,heat_thresh*heat.max())
    # Visualize the heatmap when displaying    
    heatmap = np.clip(heat, 0, 255)
    # Find final boxes from heatmap using label function
    labels = label(heatmap)
    target_list = []
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        # Define a bounding box based on min/max x and y
        bbox = np.array(((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy))))

        target_list.append(bbox)
    # return list of target cars
    return target_list, heatmap


class detector():
    def __init__(self):
        # load svc data:
        self.svc, self.X_scaler, self.orient, self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins = loadTrainedSVC("train_data.p")
        self.ystart = 400
        self.ystop = 656
        self.scale = 1.5
        self.heat_thresh = 0.4;

        
    
    def process(self,img):
        # find the cars & get box list of hits:
        box_list = find_cars(img, self.ystart, self.ystop, self.scale, self.svc, self.X_scaler, self.orient, self.pix_per_cell, self.cell_per_block, self.spatial_size, self.hist_bins)
        # get list of targets using heat map:
        targets, heatmap = filterBoxList(img,box_list, self.heat_thresh)
        # check any existing targets, and associate new targets to closest old targets
    
        output_image = img.copy()
        for box in targets:
            output_image = draw_box(output_image,box)
        
        # combine images:
        heatmap3D = stacked(heatmap)
        return np.concatenate((output_image,heatmap3D),axis=1)
       

def stacked(img):
    uimg = img.astype(np.uint8)
    # rescale:
    uimg = (img*255/img.max()).astype(np.uint8)
    return np.dstack((uimg, uimg, uimg))
    
##########################################    


# OR LOAD TRAINING:
svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,hist_bins = loadTrainedSVC("train_data.p")

ystart = 400
ystop = 656
scale = 1.5

img = mpimg.imread('test_images/test2.jpg')
  
blist = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
targets, heatmap = filterBoxList(img,blist, 0.3)

output_image = img.copy()
for box in targets:
    output_image = draw_box(output_image,box)

heatmap3D = stacked(heatmap)
testim = np.concatenate((output_image,heatmap3D),axis=1)
fig = plt.figure()
plt.imshow(testim)

fig = plt.figure()
plt.imshow(heatmap3D)

fig = plt.figure()
plt.subplot(121)
plt.imshow(img)
plt.title('Car Positions')
plt.axis('off')
plt.subplot(122)
plt.imshow(heatmap, cmap='hot')
plt.title('Heat Map')
plt.axis('off')
fig.tight_layout()

"""
SVM training function
"""
def train():
    vehicles = glob.glob('training_data/vehicles/**/*.png',recursive=True)
    non_vehicles = glob.glob('training_data/non-vehicles/**/*.png',recursive=True)
    sample_size = -1
    
    # TRAIN:
    svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,hist_bins = trainSVC(vehicles=vehicles,non_vehicles=non_vehicles,sample_size=sample_size)

"""
Video Processing tools
"""
def video_processor():
    fileName = 'test_video.mp4'
    write_output = 'output_images/' + fileName
    clip1 = VideoFileClip(fileName)
    # make car detector object:
    car_detector = detector()
    white_clip = clip1.fl_image(car_detector.process)
    white_clip.write_videofile(write_output, audio=False)
    
"""
DEMOS
"""

# DEMO 1
"""
Load and display car and non car objects:
"""
def demo1():
    vehicles = glob.glob('training_data/vehicles/**/*.png',recursive=True)
    non_vehicles = glob.glob('training_data/non-vehicles/**/*.png',recursive=True)
    car_image = mpimg.imread(np.random.choice(vehicles))
    noncar_image = mpimg.imread(np.random.choice(non_vehicles))
    f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
    f.tight_layout()

    ax1.imshow(car_image)
    ax1.set_title('Car Image', fontsize=40)
    ax1.axis('off')

    ax2.imshow(noncar_image)
    ax2.set_title('Non-car Image', fontsize=40)
    ax2.axis('off')
    plt.savefig('output_images/car_non_car.png', bbox_inches='tight')

# DEMO 2
"""
Demonstrate the response of car/ non car vechiles to features
"""
def demo2():
    vehicles = glob.glob('training_data/vehicles/**/*.png',recursive=True)
    non_vehicles = glob.glob('training_data/non-vehicles/**/*.png',recursive=True)
    car_image = mpimg.imread(np.random.choice(vehicles))
    noncar_image = mpimg.imread(np.random.choice(non_vehicles))
    # feature spaces:
    cspaces = ['YUV','RGB','HSV','HLS','YCrCb','LUV']
    fsize = 20
    for color_space in cspaces:
        orient = 9  # HOG orientations
        pix_per_cell = 8 # HOG pixels per cell
        cell_per_block = 2 # HOG cells per block
        
        # car image:
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(car_image)  
    
        # get hog features for car on channel 0:
        _, hogImgCh0 = get_hog_features(feature_image[:,:,0], orient, pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=True)
        _, hogImgCh1 = get_hog_features(feature_image[:,:,1], orient, pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=True)
        _, hogImgCh2 = get_hog_features(feature_image[:,:,2], orient, pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=True)
    
        
        f, ax = plt.subplots(1, 4, figsize=(12, 3))
        f.tight_layout()
    
        ax[0].imshow(car_image)
        ax[0].set_title('Car Image', fontsize=fsize)
    #    ax[0].axis('off')
    
        ax[1].imshow(hogImgCh0)
        ax[1].set_title('HOG Ch0', fontsize=fsize)
    #    ax[1].axis('off')
        
        ax[2].imshow(hogImgCh1)
        ax[2].set_title('HOG Ch1', fontsize=fsize)
    #    ax[2].axis('off')
        
        ax[3].imshow(hogImgCh2)
        ax[3].set_title('HOG Ch2', fontsize=fsize)
    #    ax[3].axis('off')
        plt.savefig('output_images/car_features_'+ color_space + '.png', bbox_inches='tight')
        
        # noncar image:
        if color_space != 'RGB':
            if color_space == 'HSV':
                feature_image = cv2.cvtColor(noncar_image, cv2.COLOR_RGB2HSV)
            elif color_space == 'LUV':
                feature_image = cv2.cvtColor(noncar_image, cv2.COLOR_RGB2LUV)
            elif color_space == 'HLS':
                feature_image = cv2.cvtColor(noncar_image, cv2.COLOR_RGB2HLS)
            elif color_space == 'YUV':
                feature_image = cv2.cvtColor(noncar_image, cv2.COLOR_RGB2YUV)
            elif color_space == 'YCrCb':
                feature_image = cv2.cvtColor(noncar_image, cv2.COLOR_RGB2YCrCb)
        else: feature_image = np.copy(noncar_image)  
    
        # get hog features for car on channel 0:
        _, hogImgCh0 = get_hog_features(feature_image[:,:,0], orient, pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=True)
        _, hogImgCh1 = get_hog_features(feature_image[:,:,1], orient, pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=True)
        _, hogImgCh2 = get_hog_features(feature_image[:,:,2], orient, pix_per_cell, cell_per_block, 
                            vis=True, feature_vec=True)
    
        
        f, ax = plt.subplots(1, 4, figsize=(12, 3))
        f.tight_layout()
    
        ax[0].imshow(noncar_image)
        ax[0].set_title('Non car Image', fontsize=fsize)
    #    ax[0].axis('off')
    
        ax[1].imshow(hogImgCh0)
        ax[1].set_title('HOG Ch0', fontsize=fsize)
    #    ax[1].axis('off')
        
        ax[2].imshow(hogImgCh1)
        ax[2].set_title('HOG Ch1', fontsize=fsize)
    #    ax[2].axis('off')
        
        ax[3].imshow(hogImgCh2)
        ax[3].set_title('HOG Ch2', fontsize=fsize)
    #    ax[3].axis('off')
        
        plt.savefig('output_images/noncar_features_'+ color_space + '.png', bbox_inches='tight')

# DEMO 3
"""
This demo plots the all boxes from find_cars
"""
def demo3():
    svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,hist_bins = loadTrainedSVC("train_data.p")
    ystart = 400
    ystop = 656
    scale = 1.5
    for i in range(1,7):
        img = mpimg.imread('test_images/test{}.jpg'.format(i))  
        blist = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        # plot all boxes:
        for box in blist:
            img = draw_box(img,box)
        
        fig = plt.figure()
        plt.imshow(img)
        plt.savefig('output_images/detection_{}.png'.format(i), bbox_inches='tight')

# DEMO 4
"""
this demo plots all heat maps used for filtering target boxes:
"""
def demo4():
    svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,hist_bins = loadTrainedSVC("train_data.p")
    ystart = 400
    ystop = 656
    scale = 1.5
    for i in range(1,7):
        img = mpimg.imread('test_images/test{}.jpg'.format(i))  
        blist = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        targets, heatmap = filterBoxList(img,blist, 0.3)

        # plot all boxes:
        for box in blist:
            img = draw_box(img,box)
        
        f, ax = plt.subplots(1, 2, figsize=(12, 4))
        f.tight_layout()
    
        ax[0].imshow(img)
        ax[0].set_title('Detections', fontsize=40)
    #    ax[0].axis('off')
    
        ax[1].imshow(heatmap)
        ax[1].set_title('Heatamp', fontsize=40)
        plt.savefig('output_images/heatmap_{}.png'.format(i), bbox_inches='tight')


# DEMO 5
"""
this demo plots after heatmap processing:
"""
def demo5():
    svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size,hist_bins = loadTrainedSVC("train_data.p")
    ystart = 400
    ystop = 656
    scale = 1.5
    for i in range(1,7):
        img = mpimg.imread('test_images/test{}.jpg'.format(i))  
        blist = find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins)
        targets, heatmap = filterBoxList(img,blist, 0.3)

        output_image = img.copy()
        for box in targets:
            output_image = draw_box(output_image,box)

        fir = plt.figure()
        plt.imshow(output_image)

        plt.savefig('output_images/processed_{}.png'.format(i), bbox_inches='tight')