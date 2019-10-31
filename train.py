#!/usr/bin/env python

"""Description:
The train.py is to build your CNN model, train the model, and save it for later evaluation(marking)
This is just a simple template, you feel free to change it according to your own style.
However, you must make sure:
1. Your own model is saved to the directory "model" and named as "model.h5"
2. The "test.py" must work properly with your model, this will be used by tutors for marking.
3. If you have added any extra pre-processing steps, please make sure you also implement them in "test.py" so that they can later be applied to test images.

©2019 Base code created by Yiming Peng and Bing Xue
Model implementation by Maria DaRocha
"""
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten
from tensorflow.keras import backend as K
from keras import metrics
from keras.preprocessing.image import load_img, img_to_array, array_to_img

import numpy as np
import pandas  as pd
import matplotlib.pyplot as plt
import tensorflow as tf
import random
import os
import csv
import time
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error
from skimage.io import imread, imshow
from skimage.filters import prewitt_h, prewitt_v

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
tf.set_random_seed(SEED)


def construct_model(images_and_labels):
    """
    Construct the CNN model.
    ***
        Please add your model implementation here, and don't forget compile the model
        E.g., model.compile(loss='categorical_crossentropy',
                            optimizer='sgd',
                            metrics=['accuracy'])
        NOTE, You must include 'accuracy' in as one of your metrics, which will be used for marking later.
    ***
    :return: model: the initial CNN model
    """    
    #Template: two layer dense network with Dense Layers,
    #Added: Conv2D layers and Flatten Layer

    #Sequential model: a linear stack of layers
    model = Sequential()
    
    #64 nodes in convolution layer one
    model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(300, 300, 1)))
    #32 nodes in convolution layer two
    model.add(Conv2D(32, kernel_size=3, activation='relu'))
    #kernel size: 3x3 filter matrix (can increase this if desired)
    #activation function: ReLU (Rectified Linear Activation) [as advised]
    #input shape: 300x300 (image pixels), 1 indicates 'greyscale'
    
    model.add(Flatten())
    #flatten: connection layer between convolution and dense layers
    
    model.add(Dense(units=64, activation='relu', input_dim=100))
    #units: dimension of output
    #activation: nonlinear activation function
    #input_dim: dimension of input
    
    model.add(Dense(units=3, activation='softmax'))
    #units: number of classes
    #softmax: sums outputs to 1 (translates into probabilities)
    #i.e. - "strawberry" 96%, etc.

    model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy', 'mae'])
    #loss function: categorical cross-entropy, lower score = better performance!
    #optimizer: sgd (stochastic gradient descent optimizer) 
    #metric(s): accuracy (or cat_acc), mae
    return model


def train_model(model, train_test_dataframe):
    """
    Train the CNN model
    ***
        Please add your training implementation here, including pre-processing and training
    ***
    :param model: the initial CNN model
    :return:model:   the trained CNN model
    """

    #shuffle the elements of the dataframe
    #(necessary because 'classes' are in sequential order: cherry, strawberry, tomato)
    #i.e. - avoids test sets being entirely 'tomato'
    train_test_dataframe = train_test_dataframe.sample(frac=1)

    #split the dataset manually (3/4 training, 1/4 testing)
    #roughly 75% of instances are used for training
    X_train = train_test_dataframe.loc[0:3330, 'Name']
    y_train = train_test_dataframe.loc[0:3330, 'Class'] 
    X_test = train_test_dataframe.loc[3330:, 'Name']
    y_test = train_test_dataframe.loc[3330:, 'Class'] 

    #Grayscale Processing
    for x in X_train:
        img = load_img(X_train[x], grayscale=True)
        X_train[x].append(img_to_array(img))

    for x in X_test:
        img = load_img(X_test[x], grayscale=True)
        X_test[x].append(img_to_array(img))

    #train the model using model.fit
    model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=3)
    
    return model


def save_model(model):
    """
    Save the keras model for later evaluation
    :param model: the trained CNN model
    :return:
    """
    # ***
    #   Please remove the comment to enable model save.
    #   However, it will overwrite the baseline model we provided.
    # ***
    # model.save("model/model.h5")
    print("Model Saved Successfully.")


"""
Using scikit-image library, Black & White EDA
 EXPLORING ONE CHANNEL: Greyscale values as features
 params: none
 returns: none
"""
def bw_eda():
    image = imread('file:///home/darochmari/Desktop/Final Project/Train_data/tomato/tomato_0003.jpg', as_gray=True)
    imshow(image)
    
    #check dim 
    print('Image dimensions: ', image.shape)
    
    #value matrix
    print('\nImage matrix\n', image)
    
    #All images are 300x300
    #No. of features = No. of pixels (90k)
    #Hence, we generate 1D array of length 90,000
    features = np.reshape(image, (300*300))
    
    #Shape of feature array
    print('\nFeature array dimensions: \n ', features.shape)
    
    print('\nFeature Array\n', features)

"""
Using scikit-image library, Full Color EDA
 EXPLORING THREE CHANNELS: Max RGB values as features
 params: none
 returns: none
"""
def full_color_eda():
    image = imread('file:///home/darochmari/Desktop/Final Project/Train_data/tomato/tomato_0003.jpg')
    imshow(image)
    
    #check dim 
    print('Image dimensions: ', image.shape) 
    #value matrix
    print('\nImage matrix\n', image)
    
    #All images are 300x300
    #No. of features = No. of pixels * No. of channels (270k)
    #We generate 3D matrix of size 270,000
    feature_matrix = np.zeros((300, 300)) 
    feature_matrix.shape
    
    #New matrix: Max value from RGB channels
    for i in range(0, image.shape[0]):
        for j in range(0, image.shape[1]):
            feature_matrix[i][j] = (max(int(image[i,j,0]), int(image[i,j,1]), int(image[i,j,2])))
    
    #Now produce our new 1D array of 90k features
    features = np.reshape(feature_matrix, (300*300)) 
    
    # shape of feature array
    print('\nFeature array dimensions: \n ', features.shape)    
    print('\nFeature Array\n', features)

"""
Using scikit-image library, Edge Detection EDA
    params: none
    returns: none
 
EXPLORING ONE CHANNEL: Identifying drastic changes in pixel values
 Manually, this would be done by taking the difference between 
 a selected pixel value and its adjacent pixel values. 
 
 Instead, we are able to use scikit's Prewitt kernel ([-1,0,1],[-1,0,1],[-1,0,1]).
 By multiplying the values surrounding the selected pixel against the kernel, 
 adding the values is equivalent to taking the difference.
"""
def edge_detection_eda():
    name = 'file:///home/darochmari/Desktop/Final Project/Train_data/tomato/tomato_0003.jpg'
    image = imread(name, as_gray=True)

    #horizontal edges
    prewitt_horizontal_edges = prewitt_h(image)
    #vertical edges
    prewitt_vertical_edges = prewitt_v(image)
    
    imshow(prewitt_horizontal_edges, cmap='gray')

"""
Gather all image file paths into a csv for collection & processing
Using csv and os libraries as file walkers
    ***These paths must change on school computers***
    
    params: none
    returns: none
"""
def write_images():
    cherry_path = '/home/darochmari/Desktop/Final Project/Train_data/cherry'
    strawberry_path = '/home/darochmari/Desktop/Final Project/Train_data/strawberry'
    tomato_path = '/home/darochmari/Desktop/Final Project/Train_data/tomato'
    
    with open('train_images_csv', 'w') as csvfile:
      writer = csv.writer(csvfile)
      for root, dirs, files in os.walk(cherry_path):
          for filename in files:
              writer.writerow([os.path.join(root,filename)])
      for root, dirs, files in os.walk(strawberry_path):
          for filename in files:
              writer.writerow([os.path.join(root,filename)])
      for root, dirs, files in os.walk(tomato_path):
        for filename in files:
            writer.writerow([os.path.join(root,filename)])


"""
Gather all image file paths into a csv for collection & processing
    ***This path must change on school computers***
    
    params: none
    returns: list of all file paths (csv_list)
"""
def read_images():
    with open(r"/home/darochmari/Desktop/Final Project/Template/train_images_csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        csv_list = []
        line_count = 0
        for row in csv_reader:
            csv_list.append(row)
            line_count = line_count + 1
        print("Completed line read of count: ", line_count)
    return csv_list 

"""
y_Train & y_Test label set
    Produces labels for read in images.
    params: filenames
    returns: list of labels corresponding to images (cherry, strawberry, tomato)
"""
def generate_labels(image_names):
    label_set = []
    for image in image_names:
        if("cherry" in image):
            label_set.append("cherry")
        elif("strawberry" in image):
            label_set.append("strawberry")
        else:
            label_set.append("tomato")       
    return label_set

"""
greyscale_eval
    params: file path (image_name)
    returns: 1D array [90,000,] of greyscale pixel values
"""
def greyscale_eval(image_name):
    try: 
        image = imread(image_name, as_gray=True)    
        features = np.reshape(image, (300*300))
        return features
    except:
        print("Check Picture Greyscale: ", image_name)

"""
max_rgb_eval
    params: file path (image_name)
    returns: 1D array [90,000,] of max rgb values
"""
def max_rgb_eval(image_name):
    try: 
        image = imread(image_name)    
        
        #All images are 300x300
        #No. of features = No. of pixels * No. of channels (270k)
        #Generate 3D matrix of size 270,000
        feature_matrix = np.zeros((300, 300)) 
        
        #New matrix: Max value from RGB channels
        for i in range(0, image.shape[0]):
            for j in range(0, image.shape[1]):
                feature_matrix[i][j] = (max(int(image[i,j,0]), int(image[i,j,1]), int(image[i,j,2])))
        
        #Now produce our new 1D array of 90k features
        features = np.reshape(feature_matrix, (300*300)) 
        return features
    except:
        print("Check Picture Color: ", image_name)     

"""
vert_edge_eval using prewitt kernel
    params: file path (image_name)
    returns: 1D array [90,000,] of vertical edges
"""
def vert_edge_eval(image_name):
    try: 
        image = imread(image_name, as_gray=True)    
        #vertical edges
        prewitt_vertical_edges = prewitt_v(image)        
        return prewitt_vertical_edges
    except:
        print("Unknown Error: ", image_name)  

"""
horiz_edge_eval using prewitt kernel
    params: file path (image_name)
    returns: 1D array [90,000,] of vertical edges
"""
def horiz_edge_eval(image_name):
    try: 
        image = imread(image_name, as_gray=True)    
        #horizontal edges
        prewitt_horizontal_edges = prewitt_h(image)        
        return prewitt_horizontal_edges
    except:
        print("Unknown Error: ", image_name)         
        
if __name__ == '__main__':
    
    #Exploring one data instance to gain understanding of how to proceed...   
    bw_eda()
    full_color_eda()
    edge_detection_eda()
    
    #Begin collecting images
    write_images()
    #***MUST find and replace '\' with '/'*** 
    #in VS Code before reading (if executed on Windows 10)
    unmapped_image_filenames = read_images()    
    
    #Map list of lists to a list of strings
    #Then wrap the map in a list
    image_filenames = list(map(''.join, unmapped_image_filenames))
    #print(image_filenames)
    
    #Pre-processing
    #Remove pictures with unusable pixel values
    #Requires three iterations, but uncertain as to why
    i = 0
    while (i < 3):
        for image in image_filenames:
            try:
                im = imread(image, as_gray=True)
                f = np.reshape(im, (300*300))
            except:
                image_filenames.remove(image)
                print("Image Removed:  ", image)
        print(len(image_filenames))
        i = i+1
    #remove black & white filter instances to avoid noise
    image_filenames.remove("/home/darochmari/Desktop/Final Project/Train_data/strawberry/strawberry_1581.jpg")

    print(len(image_filenames)) #desired: 4439
    
    #"Answer Key" for this training set
    label_set = generate_labels(image_filenames)
    print("Labels Generated")

    """
    **** KERAS MODEL CONSTRUCTION ***
    Construct dataframe as keras sees it
    """
    k_data = {'Name': image_filenames,
              'Class': label_set}
    keras_df = pd.DataFrame(k_data)

    # construct the model
    model = construct_model()

    # use the intended [keras] dataframe to train the model
    model = train_model(model, keras_df)

    # save_model(model)

    """
    **** Manual Investigation: Convolution & Pooling ****
    Convolution: Grayscale, Prewitt Edge Detection (Vert & Horiz)
    Pooling: Max_RGB
    """
    #Feature ONE: Greyscale
    train_greyscale_features = [] 
    for image in image_filenames:        
        train_greyscale_features.append(greyscale_eval(image))
    print("Greyscale Complete")

    
    #Feature TWO: Color (Max RGB)
    train_color_features = []
    for image in image_filenames:        
        train_color_features.append(max_rgb_eval(image))
    print("Max RGB Complete")

    
    #Feature THREE: Vertical Edge Detection
    vert_edge_features = []
    for image in image_filenames:        
        vert_edge_features.append(vert_edge_eval(image))
    print("Prewitt Vertical Edges Complete")
    
    #Feature FOUR: Horizontal Edge Detection
    horiz_edge_features = []
    for image in image_filenames:        
        horiz_edge_features.append(horiz_edge_eval(image))    
    print("Prewitt Horizontal Edges Complete")
    
    #intialise data of lists
    data = {'Name':image_filenames, 
            'Greyscale':train_greyscale_features,
            'Max_RGB':train_color_features,
            'V_Edges':vert_edge_features,
            'H_Edges':horiz_edge_features,
            'Class':label_set}
    #ensure list lengths are uniform, then
    #create dataframe from ALL collected features
    print(len(image_filenames))
    print(len(label_set))
    print(len(train_greyscale_features))
    print(len(train_color_features))
    print(len(vert_edge_features))
    print(len(horiz_edge_features))

    base_dataframe = pd.DataFrame(data)
    
    #shuffle the elements of the dataframe
    #(necessary because 'classes' are in sequential order: cherry, strawberry, tomato)
    #i.e. - avoids test sets being entirely 'tomato'
    base_dataframe = base_dataframe.sample(frac=1)
    print(len(base_dataframe))

    #display head of base df
    print(base_dataframe.head)

    """
    #split the dataset manually (3/4 training, 1/4 testing)
    #roughly 75% of instances are used for training
    X_train = base_dataframe.loc[0:3330, 1:4]
    #print(X_train)
    y_train = base_dataframe.loc[0:3330, 5]
    #print(y_train)
    X_test = base_dataframe.loc[3330:, 1:4]
    y_test = base_dataframe.loc[3330:, 5]
    """
    """
    **** BASIC MULTI-LAYERED PRECEPTRON (MLP) ****
    Time elapsed: ? seconds
    hidden_layer_sizes = 100   
    activation = ‘relu’, default, the rectified linear unit function, 
                    returns f(x) = max(0, x)
    solver = Default solver ‘adam’ performs well on large datasets, 
                in terms of both training time and validation score.       
    learning_rate = 'constant'
    max_iter = 1000
                Maximum number of iterations. The solver iterates until convergence 
                (determined by ‘tol’) or this number of iterations. For stochastic 
                solvers (‘sgd’, ‘adam’), note that this determines the number of epochs 
                (how many times each data point will be used), not the number of gradient steps.
    learning_rate_init = 0.001, default
                        The initial learning rate used. It controls the step-size in 
                        updating the weights. [Only used when solver=’sgd’ or ‘adam’]
    Reference: sklearn.neural_network.MLPRegressor Documentation
    """
    """
    start_time = time.time()
    
    mlpr = MLPRegressor(hidden_layer_sizes=(5,), #only a small number of features
                                           activation='relu', #default
                                           solver='adam', 
                                           learning_rate='constant', #default
                                           max_iter=1000,
                                           learning_rate_init=0.001) #default
    model = mlpr.fit(X_train, y_train.ravel())
    
    print("--- %s seconds ---" % (time.time() - start_time))
    
    #store model statistics
    MLPR_model = ['MLP Reg', (time.time() - start_time)]
    
    #Predict values
    Y_pred = model.predict(X_test)    
    #MSE
    MLPR_model.append(mean_squared_error(y_test, Y_pred))    
    #RMSE
    MLPR_model.append(np.sqrt(mean_squared_error(y_test, Y_pred)))    
    #MAE
    MLPR_model.append(mean_absolute_error(y_test, Y_pred))    
    #R-Squared Score
    MLPR_model.append(r2_score(y_test, Y_pred))    
    print("ML Perceptron Reg Model Statistics:", MLPR_model)
    """

    


