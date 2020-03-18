# Import model which has a class that generates a CNN model, and also have some helper functions for model saving and loading
from util import model
from util import dataLoader #dataLoader loads the images directly from URL on the fly, done ony once and is then cached!

import os
import numpy as np
import matplotlib.pylab as plt

import tensorflow as tf
from sklearn.model_selection import train_test_split

def train(args):
    
    # Generate Data Loader
    dataloader = dataLoader.DataLoader(args)
    (X_train, y_train), (X_test, y_test) = dataloader.generateData()
    
    # Perform data Augmentation
    X_train_shaped, y_train_shaped = dataloader.preprocess_data(
                                            X_train, y_train, 
                                            use_augmentation=True, 
                                            nb_of_augmentation=args.num_aug
                                            )
    X_test_shaped, y_test_shaped   = dataloader.preprocess_data(X_test,  y_test)


    # Now split the training data into train and Validation, helpful to monitor overfitting during training of models
    X_train_, X_val_, y_train_, y_val_ = train_test_split(X_train_shaped, y_train_shaped,
                                                            test_size=0.2, random_state=42)
    
    # Saving the best checkpoint for each iteration
    filepath = os.path.join(args.save_dir, "fashion_mnist-0.hdf5")
    
    # Create CNN model to train
    print("Generating CNN Model ....")
    model_obj = model.GenerateModel(args)
    cnn_model = model_obj.cnn_model(modelArchitecture=args.model_arch)
    
    # Fit the dataset for training
    history = cnn_model.fit(X_train_, y_train_,
                                batch_size=args.batch_size,
                                epochs=args.num_epochs,
                                verbose=1,
                                validation_data=(X_val_, y_val_),
                                callbacks=[
                                    tf.keras.callbacks.ModelCheckpoint(filepath, monitor='val_loss', verbose=1, save_best_only=True)
                                ]
                            )

    
    print("Training done dumping the final model and the graph now..")
    print("save_dir -->", args.save_dir)
    model_obj.saveModel(cnn_model)

    print("Model SAved, Please go through, provided Jupyter notebooks to see nice plots..")