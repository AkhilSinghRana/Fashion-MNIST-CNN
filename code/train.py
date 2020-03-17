# Import model which has a class that generates a CNN model, and also have some helper functions for model saving and loading
from util import model
from util import dataLoader #dataLoader loads the images directly from pandas dataframe on the fly

import numpy as np
import matplotlib.pylab as plt

def train(args):
    
    # Generate Data Loader
    dataloader = dataLoader.DataLoader(args)
    train_data_gen, val_data_gen = dataloader.dataGenerator()

    # Create CNN model to train
    print("Generating CNN Model ....")
    model_obj = model.GenerateModel(args, dataloader.num_classes)
    cnn_model = model_obj.cnn_model()
    print("Number of training images ", dataloader.num_train_images)
    # Fit the dataset for training
    history = cnn_model.fit_generator(
                    train_data_gen,
                    steps_per_epoch=dataloader.num_train_images // args.batch_size,
                    epochs=args.num_epochs,
                    validation_data=val_data_gen,
                    validation_steps=dataloader.num_val_images // args.batch_size
                )

    
    print("Training done dumping the model now..")
    print("save_dir -->", args.save_dir)
    model_obj.saveModel(cnn_model)
