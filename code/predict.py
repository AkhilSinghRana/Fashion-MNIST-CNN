
from util import dataLoader #dataLoader loads the images directly from directory on the fly for prediction
from util import model
from skimage import io
from util import preprocessing
import numpy as np
from matplotlib import pyplot as plt
def checkImage(image):
        
        raw_image = io.imread(image) #read the image to numpy array
        # Check if number of channels is 3 or more
        raw_image_dim =  raw_image.ndim
        if raw_image_dim > 2:
            gray_image = preprocessing.convertToGray(image)
        
        else:
            print("Image is already single channel")
            gray_image = raw_image
        
        
        return gray_image

# This function will plot images in the form of a grid with 1 row and 5 columns where images are placed in each column.
def plotImage(image, title="Unknown"):
    plt.imshow(image[0,:,:,0], cmap=plt.cm.gray)
    plt.title(title)

    plt.show()

def predict(args):
    print("Loading the model for inference")
    
    model_obj = model.GenerateModel(args, num_classes=3)
    loaded_model = model_obj.loadModel()
    loaded_model.summary()
    
    if args.predict_mode == "single":
        #Step1: Convert to Gray if not already gray!
        gray_image = checkImage(args.predict_image) 
        
        #Step2: Preprocessing, make image in the range of [0,1], this is reuired because my CNN model was trained under this setting!
        gray_image = (gray_image/255)
        #gray_image = np.around(gray_image, 2)
        
        
        #Step3: Add the image to a batch where it's the only member, this is a requirement for our model
        img = (np.expand_dims(gray_image,2))
        img = (np.expand_dims(img,0))
        
        #Step4: Model Preditcion, pass the image to the model for predictions and store them in results
        result = loaded_model.predict(img)
        result_index = np.argmax(result) #We do this because our model gives the probability for each class
        
        #Define Class names for mapping the indexes
        class_names = ["FULL_VISIBILITY", "PARTIAL_VISIBILITY", "NO_VISIBILITY"]
        print("Our model thinks that this image has --> ", class_names[result_index])
        
        #Final Step: Let's plot the results, because we love ploting!
        plotImage(img, class_names[result_index])

    else:
        print("We have support for single image prediction at the moment!, check Jupyter notebook provided to see how batch predictions can be done.")
        raise Exception