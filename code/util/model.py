######################
#   CNN model Architectures
######################
# We use tf 2.1 and tf.Keras API

## Import packages
import tensorflow as tf

import tensorflow.keras as keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, Activation, BatchNormalization
import tensorflow.keras.optimizers as optimizers


#Load data generator and preprocessing module
from tensorflow.keras.preprocessing.image import ImageDataGenerator


#import other modules
import os
import numpy as np
import matplotlib.pyplot as plt

class GenerateModel():
    
    # Initialize the class, with some default variables
    def __init__(self, args=None):
        """
            args = Argument parser from the options.py file
        """
        self.args = args

    # This function defines the CNN architecture
    def cnn_model(self, modelArchitecture="custom"):

        model = tf.keras.Sequential()
        
        if self.args.model_arch=="custom":
            """
                The customized model is selected after finetuning it many time to reach the best results for this problem
            """
            
            model.add(tf.keras.layers.InputLayer(input_shape=(self.args.img_h, self.args.img_w, self.args.num_channels)))
            # Normalization
            model.add(tf.keras.layers.BatchNormalization())
            
            # Conv + Maxpooling
            model.add(tf.keras.layers.Convolution2D(64, (4, 4), padding='same', activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

            # Dropout
            model.add(tf.keras.layers.Dropout(0.1))
            
            # Conv + Maxpooling
            model.add(tf.keras.layers.Convolution2D(64, (4, 4), activation='relu'))
            model.add(tf.keras.layers.MaxPooling2D(pool_size=(2, 2)))

            # Dropout
            model.add(tf.keras.layers.Dropout(0.3))

            # Converting 3D feature to 1D feature Vector
            model.add(tf.keras.layers.Flatten())

            # Fully Connected Layer
            model.add(tf.keras.layers.Dense(256, activation='relu'))

            # Dropout
            model.add(tf.keras.layers.Dropout(0.5))
            
            # Fully Connected Layer
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            
            # Normalization
            model.add(tf.keras.layers.BatchNormalization())

            model.add(tf.keras.layers.Dense(self.args.num_classes, activation='softmax'))
        
        elif self.args.model_arch=="vgg_like":
            # first CONV => RELU => CONV => RELU => POOL layer set
            model.add(Conv2D(32, (3, 3), padding="same",input_shape=(self.args.img_h,self.args.img_w,self.args.num_channels)))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=3))
            model.add(Conv2D(32, (3, 3), padding="same"))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=3))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            # second CONV => RELU => CONV => RELU => POOL layer set
            model.add(Conv2D(64, (3, 3), padding="same"))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=3))
            model.add(Conv2D(64, (3, 3), padding="same"))
            model.add(Activation("relu"))
            model.add(BatchNormalization(axis=3))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            # first (and only) set of FC => RELU layers
            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation("relu"))
            model.add(BatchNormalization())
            model.add(Dropout(0.5))

            # softmax classifier
            model.add(Dense(self.args.num_classes))
            model.add(Activation("softmax"))

        elif self.args.model_arch=="v1":
            """
                first version
            """
            print("Version 1")
            model = Sequential()
            model.add(Conv2D(16, (3, 3), padding='same',
                            input_shape=(self.args.img_h, self.args.img_w, self.args.num_channels)))
            model.add(Activation('relu'))
            model.add(Conv2D(32, (3, 3)))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))


            model.add(Conv2D(64, (3, 3)))
            model.add(BatchNormalization())
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(0.25))

            model.add(Flatten())
            model.add(Dense(512))
            model.add(Activation('relu'))
            model.add(Dropout(0.5))
            model.add(Dense(self.args.num_classes, activation='softmax'))
        else:
            raise NotImplementedError

        """
            ################################
            Compile the model now with different Optimizers, again flexibility to use either adam or RMSProp is given
        """
        # Model compile defines the otimizer and loss function to choose
        if self.args.optimizer=="RMS":
            model.compile(optimizers.RMSprop(lr=0.0005, decay=1e-6),
                            loss=tf.keras.losses.categorical_crossentropy,
                            metrics=["accuracy"])
        elif self.args.optimizer=="Adam":
            model.compile(optimizer='adam', 
                            loss=tf.keras.losses.categorical_crossentropy, 
                            metrics=['accuracy'])
        else:
            print("Optimizer not supported yet check your model.py for valid optimizers")
            raise NotImplementedError
        
        return model

    def model_summary(self, model=None):
        model.summary()

    def saveModel(self, model):
        # Let's save this model now
        save_dir = os.path.join(self.args.save_dir, self.args.exp_name)

        tf.keras.models.save_model(model, save_dir)

    def loadModel(self, model_path=None):
        model_dir = os.path.join(self.args.load_dir, self.args.exp_name) if model_path is None else model_path
        loaded_model = tf.keras.models.load_model(model_dir)

        print("Model Details!")
        loaded_model.summary()
        return loaded_model