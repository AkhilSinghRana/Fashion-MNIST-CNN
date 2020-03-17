import os, glob, random
from tensorflow.keras.preprocessing import image as image_preprocessing
import pandas as pd
from skimage import io
from sklearn.model_selection import train_test_split

from util import preprocessing

class DataLoader():
    def __init__(self, args=None):
        print("Initializing DataLoader ...")
        self.args = args
        self.num_classes = 0 # Number of class to train on
        self.num_train_images = 0 # Number of training samples
        self.num_val_images = 0 # Number of validation samples
        self.num_test_images = 0 # number of test samples
    
    def checkDataValidity(self, image_dir):
        #check if the images passed are GrayScale, if not let's preprocess them first
        images = glob.glob(image_dir+"/*.png")
        
        random_image = images[random.randint(0, len(images)-1)] # Get a randome image from the directory
        raw_image = io.imread(random_image) #read the image to numpy array
        # Check if number of channels is 3 or more
        raw_image_dim =  raw_image.ndim
        if raw_image_dim > 2:
            save_dir = os.path.join(self.args.input_dir, "GrayImages")
            if not os .path.exists(save_dir):
                os.makedirs(save_dir)
            for image in images:                
                gray_image = preprocessing.convertToGray(image)
                image_name = image.split("/")[-1]
                print("converting ",image_name, " to Gray Image")
                
                io.imsave(save_dir+"/"+image_name, gray_image)
            image_dir = save_dir
        
        return image_dir

    def dataGenerator(self):
        print("Looking for the data at -->", self.args.input_dir)
        
        image_dir = os.path.join(self.args.input_dir , self.args.images_dir)
        image_dir = self.checkDataValidity(image_dir) #Check if the images are gray scale
        csv_filename  = os.path.join(self.args.input_dir, self.args.csv_filename)

        #Read the main CSV
        data = pd.read_csv(csv_filename)
        # Do a Split for train and validation
        train, val = train_test_split(data, test_size=0.2) # 80-20 split
        self.num_classes = len(data[" LABEL"].value_counts())
        self.num_train_images = len(train["IMAGE_FILENAME"].value_counts())
        self.num_test_images = len(val["IMAGE_FILENAME"].value_counts())
        self.num_val_images = self.num_test_images
        

        train_csv = train.to_csv (os.path.join(self.args.input_dir, "train.csv"), index = None, header=True) #Don't forget to add '.csv' at the end of the path
        val_csv = val.to_csv (os.path.join(self.args.input_dir, "test.csv"), index = None, header=True) #Don't forget to add '.csv' at the end of the path

        print(" Number of Classes : {} -----------".format(self.num_classes))
        
        if self.args.mode == "train":
            # Create a data generator that generates the data on the fly with data augmentation
            train_image_gen = image_preprocessing.ImageDataGenerator(rescale=1./255, horizontal_flip= True)
            val_image_gen = image_preprocessing.ImageDataGenerator(rescale=1./255)
            
            # Data Generator with specific batch size from dataframe
            train_data_gen = train_image_gen.flow_from_dataframe(train,
                                                            directory=image_dir,
                                                            x_col='IMAGE_FILENAME',
                                                            y_col=' LABEL',
                                                            target_size=(self.args.img_h, self.args.img_h),
                                                            color_mode='grayscale',
                                                            Classes = ["FULL_VISIBILITY", "PARTIAL_VISIBILITY", "NO_VISIBILITY"],
                                                            class_mode='categorical',
                                                            batch_size=self.args.batch_size,
                                                            shuffle=True,
                                                            interpolation='nearest',
                                                            validate_filenames=True)
                                        
            val_data_gen = val_image_gen.flow_from_dataframe(val,
                                                            directory=image_dir,
                                                            x_col='IMAGE_FILENAME',
                                                            y_col=' LABEL',
                                                            weight_col=None,
                                                            target_size=(self.args.img_h, self.args.img_h),
                                                            color_mode='grayscale',
                                                            Classes = ["FULL_VISIBILITY", "PARTIAL_VISIBILITY", "NO_VISIBILITY"],
                                                            class_mode='categorical',
                                                            batch_size=self.args.batch_size,
                                                            shuffle=True,
                                                            interpolation='nearest',
                                                            validate_filenames=True)

            
            return train_data_gen, val_data_gen
        
        elif self.args.mode == "predict":
            
            test_image_gen = image_preprocessing.ImageDataGenerator(rescale=1./255)

            #Create test data gen
            test_data_gen = test_image_gen.flow_from_directory(directory=self.args.image_dir,
                                                            x_col='IMAGE_FILENAME',
                                                            y_col=' LABEL',
                                                            weight_col=None,
                                                            target_size=(self.args.img_h, self.args.img_h),
                                                            color_mode='grayscale',
                                                            class_mode='categorical',
                                                            batch_size=self.args.batch_size,
                                                            shuffle=True,
                                                            interpolation='nearest',
                                                            validate_filenames=True)

            return test_data_gen

        else:
            raise NotImplementedError