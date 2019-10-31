# Image-Classification-with-Keras
Final Project for COMP309: Machine Learning Tools and Techniques 

Maria DaRocha, Oct. 2019
___________________________________________________________
Summary:

  The penultimate goal of a Convolutional Neural Network (CNN) is for its layers to connect in such a way that the model learns features through convolution, introduces non-linearity through activation functions, reduces dimensionality (while preserving partial invariance) through pooling, produces high-level features, and then uses dense layers to return an image’s likelihood (probability) of belonging to a particular class. It is a form of deep learning in machines.
___________________________________________________________
Outcome:

A sequential CNN that distinguishes test data between digital images of: cherrys, strawberrys, and tomatoes with a 73.3% accuracy.
- Language: Python
- Library Dependencies: Keras, Tensorflow, Sklearn, Scikit-image (skimage), Numpy, Pandas, Random, OS, CSV, Time, (Tensorboard: Recommended)
___________________________________________________________

*If you're reading this, please be aware that credit for the creation of the original assignment, a portion of the base code, and the training/testing datasets is given to Victoria University of Wellington (VUW).*
  
The goal of this assignment was to:

  - Produce a Sequential CNN that distinguishes between digital images of: cherrys, strawberrys, and tomatoes with a degree of accuracy which surpasses random guessing (roughly 33% acc).
  
  - Gain a deeper understanding of the development process for CNNs, including [layers]: 
          - Convolution 
          - Pooling 
          - Dropout 
          - Fully-connected (Dense/MLP)
          - Activation, and 
          - Normalization

  - To understand the basic image Classification pipeline.
  
  - To properly use available data for model training and hyper-parameter tuning.
  
  - Be able to implement CNNs using Keras.
  
  - Be able to save a trained CNN model after the training process.
  
  - Be able to load a trained/saved CNN model to classify unseen test data.
  
  - To understand the influences of different loss functions and optimisers for training CNNs.
  
  - To learn and be able to use techniques to improve the CNN performance, e.g., feature engineering and transfer learning.
  
  - To write a clear report on the model & subject

