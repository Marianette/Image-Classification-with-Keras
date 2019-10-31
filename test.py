#!/usr/bin/env python

"""Description:
The test.py is to evaluate your model on the test images.
***Please make sure this file work properly in your final submission***

©2019 Base code created by Yiming Peng and Bing Xue
Preprocessing edits created by Maria DaRocha
"""
from tensorflow.python.keras.models import load_model
from tensorflow.python.keras.preprocessing.image import img_to_array, ImageDataGenerator

# You need to install "imutils" lib by the following command:
#               pip install imutils
from imutils import paths
from sklearn.preprocessing import LabelBinarizer
import cv2
import os
import argparse
import csv
import numpy as np
import random
import tensorflow as tf
from skimage.io import imread, imshow
from skimage.filters import prewitt_h, prewitt_v

# Set random seeds to ensure the reproducible results
SEED = 309
np.random.seed(SEED)
random.seed(SEED)
#tf.set_random_seed(SEED)


def parse_args():
    """
    Pass arguments via command line
    :return: args: parsed args
    """
    # Parse the arguments, please do not change
    args = argparse.ArgumentParser()
    args.add_argument("--test_data_dir", default = "data/test",
                      help = "path to test_data_dir")
    args = vars(args.parse_args())
    return args


def load_images(test_data_dir, image_size = (300, 300)):
    """
    Load images from local directory
    :return: the image list (encoded as an array)
    """
    # loop over the input images
    images_data = []
    labels = []
    imagePaths = sorted(list(paths.list_images(test_data_dir)))
    for imagePath in imagePaths:
        # load the image, pre-process it, and store it in the data list
        image = cv2.imread(imagePath)
        image = cv2.resize(image, image_size)
        image = img_to_array(image)
        images_data.append(image)

        # extract the class label from the image path and update the
        # labels list
        label = imagePath.split(os.path.sep)[-2]
        labels.append(label)
    return images_data, sorted(labels)


def convert_img_to_array(images, labels):
    # Convert to numpy and do constant normalize
    X_test = np.array(images, dtype = "float") / 255.0
    y_test = np.array(labels)

    # Binarize the labels
    lb = LabelBinarizer()
    y_test = lb.fit_transform(y_test)

    return X_test, y_test


def preprocess_data(X):
    """
    Pre-process the test data.
    :param X: the original data
    :return: the preprocess data
    """
    # NOTE: # If you have conducted any pre-processing on the image,
    # please implement this function to apply onto test images.
    return X


def evaluate(X_test, y_test):
    """
    Evaluation on test images
    ******Please do not change this function******
    :param X_test: test images
    :param y_test: test labels
    :return: the accuracy
    """
    # batch size is 16 for evaluation
    batch_size = 16

    # Load Model
    model = load_model('model/model.h5')
    return model.evaluate(X_test, y_test, batch_size, verbose = 1)


"""
Gather all image file paths into a csv for collection & processing
Using csv and os libraries as file walkers
    ***These paths must change on school computers***
    
    params: none
    returns: none
"""
def write_images():
    cherry_path = '/home/darochmari/Desktop/Final Project/Test/cherry'
    strawberry_path = '/home/darochmari/Desktop/Final Project/Test/strawberry'
    tomato_path = '/home/darochmari/Desktop/Final Project/Test/tomato'
    
    with open('test_images_csv', 'w') as csvfile:
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
    with open(r"/home/darochmari/Desktop/Final Project/Template/test_images_csv", "r") as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\n')
        csv_list = []
        line_count = 0
        for row in csv_reader:
            csv_list.append(row)
            line_count = line_count + 1
        print("Completed line read of count: ", line_count)
    return csv_list 

if __name__ == '__main__':
    # Parse the arguments
    args = parse_args()

    # Test folder
    test_data_dir = args["test_data_dir"]

    #Begin collecting images
    write_images()
    #***MUST find and replace '\' with '/'*** 
    #in VS Code before reading  
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

    # Image size, please define according to your settings when training your model.
    image_size = (64, 64)

    # Load images
    images, labels = load_images(test_data_dir, image_size)

    # Convert images to numpy arrays (images are normalized with constant 255.0), and binarize categorical labels
    X_test, y_test = convert_img_to_array(images, labels)

    # Preprocess data.
    # ***If you have any preprocess, please re-implement the function "preprocess_data"; otherwise, you can skip this***
    X_test = preprocess_data(X_test)

    # Evaluation, please make sure that your training model uses "accuracy" as metrics, i.e., metrics=['accuracy']
    loss, accuracy = evaluate(X_test, y_test)
    print("loss={}, accuracy={}".format(loss, accuracy))
