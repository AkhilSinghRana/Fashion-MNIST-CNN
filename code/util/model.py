######################
#   CNN model Architecture
######################
# We use tf 1.15 and tf.Keras API

## Import packages
import pandas as pd
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
    def __init__(self, args=None, num_classes=0):
        """
            args = Argument parser from the options.py file
        """
        self.args = args
        self.num_classes = num_classes

    # This function defines the CNN architecture
    def cnn_model(self):

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
        
        model.add(Dense(self.num_classes, activation='softmax'))

        
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
        
        # Generate the model Summary
        print("Model Details!")
        model.summary()
        return model

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