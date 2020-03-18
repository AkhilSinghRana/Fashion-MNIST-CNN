"""
All Data preprocessing and augmentation is done here
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

class DataLoader():
    """
        A class for data loading and preprocessing 
    """
    def __init__(self, args=None):
        self.args=args

        """
        # Define the options for augmentation
        """
        self.datagen = ImageDataGenerator(
            rotation_range=10,
            horizontal_flip=True,
            fill_mode='nearest'
        )
    def generateData(self):
        fashion_mnist = keras.datasets.fashion_mnist
        (X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

        print("Train Samples:", len(X_train))
        print("Test Samples:",  len(X_test))

        return (X_train, y_train), (X_test, y_test)


    def image_augmentation(self,image, nb_of_augmentation):
        '''
        Generates new images bei augmentation
        image : raw image
        nb_augmentation: number of augmentations
        images: array with new images
        '''
        images = []
        image = image.reshape(1, self.args.img_h, self.args.img_w, self.args.num_channels)
        i = 0
        for x_batch in self.datagen.flow(image, batch_size=1):
            images.append(x_batch)
            i += 1
            if i >= nb_of_augmentation:
                # interrupt augmentation
                break
        return images

    def preprocess_data(self, images, targets, use_augmentation=False, nb_of_augmentation=1):
        """
        images: raw image
        targets: target label
        use_augmentation: True if augmentation should be used
        nb_of_augmentation: If use_augmentation=True, number of augmentations
        """
        print("Augmenting images...")
        X = []
        y = []
        for x_, y_ in zip(images, targets):
            
            # scaling pixels between 0.0-1.0
            x_ = x_ / 255.
            x_ = x_.reshape(self.args.img_h, self.args.img_w, self.args.num_channels)
            # data Augmentation
            if use_augmentation:
                argu_img = self.image_augmentation(x_, nb_of_augmentation)
                for a in argu_img:
                    reshaped_a = a.reshape(self.args.img_h, self.args.img_w, self.args.num_channels)
                    X.append(reshaped_a)
                    y.append(y_)
                    
                    
            X.append(x_)
            y.append(y_)
        print('*Preprocessing completed: %i samples\n' % len(X))
        return np.array(X), tf.keras.utils.to_categorical(y)


if __name__ == "__main__":
    args = options.parseArguments()
    print("Call the script from train or predict.py!")
    

    
    
