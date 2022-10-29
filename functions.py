import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import keras

import tensorflow as tf

from keras import layers,regularizers, callbacks, utils, applications, optimizers

from keras.models import Sequential, Model, load_model

import pickle

#from sklearn.model_selection import train_test_split

from tensorflow.keras import models, layers
from tensorflow.keras.layers import Input, Flatten, Conv2D, LeakyReLU, GlobalAveragePooling2D, MaxPooling2D, Dropout, Dense
from tensorflow.keras.preprocessing.image import array_to_img, img_to_array, load_img

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.applications.resnet50 import ResNet50

def prepare_x_y(train):
    """
    Function which divide the training data into X (=image data) and y (=labels)
    
    INPUT 
        * train: dataframe with shape(m, 31)
    OUTPUTS
        * X : numpy.array containing image data, shape=(m, 96, 96, 1)
        * y : numpy array containing labels, shape=(m, 30)
    """
    
    imgs = []
    points = []
    
    m = train.shape[0] # size of the images / number of pixels 
    
    X_tr = train['Image'] # X part
    y_tr = train.drop('Image',axis = 1) # y part
    
    # converting the image [int int int ... int int] into an 2D array [[int, ...,int][...] ... [...]] 
    for i in range(m):
        img = X_tr.iloc[i] # for each pixel 
        img = img.split(' ') # spaces are the delimiters 
        imgs.append(img) 
        
        point = y_tr.iloc[i,:] 
        points.append(point)

    X_train = np.array(imgs, dtype = 'float') # to array
    X_train = X_train.reshape(-1,96,96,1) # reshape into 2D array 
    y_train = np.array(points, dtype = 'float') # to array
    X_train = X_train / 255.0 # normalization : from [0 ... 255] range of values to [0,1] range 
    
    return X_train, y_train

def load_data():
    train_data = pd.read_csv("./data/training_0.csv")

    for i in range(9):
        df = pd.read_csv("./data/training_" + str(i+1) + ".csv")
        train_data = pd.concat([train_data, df])

    test_data = pd.read_csv("./data/test_0.csv")

    for i in range(2):
        df_test = pd.read_csv("./data/test_" + str(i+1) + ".csv")
        test_data = pd.concat([test_data, df])
    
    lookId_data = pd.read_csv('./data/IdLookupTable.csv')

    return train_data, test_data, lookId_data

def get_images(images, points, nbColumns, shrinkage=0.2, fileName='monFichier.png'):
    """
    Function to plot images and the corresponding facial keypoints.

    INPUTS:
      * images    : numpy array with shape (m, d, d, c), dtype=float
      * points    : numpy array with shape (m,), dtype=float
      * nbColumns : number of columns in the resulting image grid
      * shrinkage : how much each image to be shrinked for display
    """

    nbIndex, height, width, intensity = images.shape
    nbRows = nbIndex//nbColumns
    print("------------------")
    print(f"Number of rows: {nbRows}, number of cols: {nbColumns}")

    fig_width = int(width*nbColumns*shrinkage)
    fig_height = int(height*nbRows*shrinkage)

    fig, axes = plt.subplots(nbRows, nbColumns, 
                            figsize=(fig_width, fig_height))

    print(f"Figure width: {fig_width}, height: {fig_height}")
    print("------------------")
    axes = axes.flatten()

    for k in range(nbIndex):
        img = images[k]
        img = array_to_img(img)
        ax = axes[k]
        ax.imshow(img, cmap="Greys_r")
        pnt_x = [points[k][2*j] for j in range(15)]
        pnt_y = [points[k][2*j+1] for j in range(15)]
        ax.scatter(pnt_x,pnt_y,s=200,c='r')
        ax.set_axis_off()

    plt.tight_layout()
    plt.savefig(fileName)

def create_model():
    model = tf.keras.models.Sequential()
    pretrained_model = ResNet50(input_shape=(96,96,3), include_top=False, weights='imagenet')
    pretrained_model.trainable = True

    model.add(Conv2D(3, (1,1), padding='same', input_shape=(96,96,1)))
    model.add(LeakyReLU(alpha=0.1))
    model.add(pretrained_model)
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(1 - dropout))
    model.add(Dense(30))
    model.summary()

    return model