#Import required packages

from skimage import io
import numpy as np

# Import options
import options



def convertToGray(image):
    #skimage
    original_image = io.imread(image) # image is stored just for visualization purposes
    #Numpy
    np_image = np.asarray(original_image)
    # Now Let's try a different formula, to keep most out of the Blue-Channel, 
        #Y = 0.04 R + 0.18 G + 0.99 B.
    gray_Corrected_image = np.dot(np_image[...,:3], [0.04, 0.18, 0.98])

    return gray_Corrected_image


if __name__ == "__main__":
    args = options.parseArguments()
    print("Call the script from train or predict.py!")
    

    
    
