***

# Classifying the visibility of ID cards in photos

Note*: There is a pre compiled README.html provided with this solution, for more appealing documentaiton(with images!).

The folder images inside data contains several different types of ID documents taken in different conditions and backgrounds. The goal is to use the images stored in this folder and to design an algorithm that identifies the visibility of the card on the photo (FULL_VISIBILITY, PARTIAL_VISIBILITY, NO_VISIBILITY).

## Data

Inside the data folder you can find the following:

### 1) Folder images
A folder containing the challenge images.

### 2) gicsd_labels.csv
A CSV file mapping each challenge image with its correct label.
	- **IMAGE_FILENAME**: The filename of each image.
	- **LABEL**: The label of each image, which can be one of these values: FULL_VISIBILITY, PARTIAL_VISIBILITY or NO_VISIBILITY. 


## Dependencies:
Note*: There is a pre compiled README.html provided with this solution, for more appealing documentaiton(with images!). Ignore if already looking at HTML version.

	Dependencies of this project is written in requirements.txt file.
Main Dependencies:
- Jupyterlab
- scikit-learn
- tensorflow [CPU/GPU], according to your machine, I have run this entire project on my notebook with Quadro 3000 GPU.
- numpy
- scikit-image

Proposed Optionals:
- Ubuntu OS
- python3.7.5 or above
- virtualenv
- pip 20.0.2 or above

## Run Instructions

	This project was run under an Ubuntu environment, this is not a dependency, but Windows machine was not tested(It would most probably still work on windows). Feel free to contact me for any troubleshooting.

Proposed steps for a virtualenvironment Setup:

```shell
virtualenv env_name -p python3 

source env_name/bin/activate

pip install -e .
```

Now you should be ready to start a training or predict the trained model on any image(RGB/Gray) both are automatically handeled by the code/predict.py code!	

Sample Train command:
	
- python code/main.py --mode train

	The images_dir flag can be given and is only needed to setup the path variables required for dataLoader. Default path is pointing towards pre-precessed GrayImages, but you can also pass "images" as a path and then dataloader will handle preprocessing automatically. The data will not be loaded directly from the folder! train.py will read the csv file into a pandas dataframe and seperate it into train and test pandas dataframe, which will then be used to load images on the fly using tensorflowImageGenerator, more details in code/util/dataLoader.py or the notebooks provided. Also, train, test csv created during this process gets saved in the data folder automatically, just for reference purposes.


Sample Prediction command:

-	python code/main.py --mode predict --load_dir artifacts --exp_name Experiment_1 --predict_image data/images/GICSD_5_7_233.png

	Trained model is provided with the solution zip for predicting images directly!

	--load_dir = directory where the model is saved after training

	--exp_name = name of the experiment that was mentioned during training defaults to Experiment_1. This is also taken as the name of the model while saving it.
	
	--predict_image = Path to the image #This can be both "either RGB or Gray Scale image". Predict.py makes sure that the proper preprocessing and inference pipelines are run to this image before getting the model predictions.

	The project was coded to provide maximum flexibilty, therefore, you can check more parameters that can be tweeked in code/options.py or simply run 
	
	"python code/main.py --help"


## Approach

There, were many options that could have been tried but this project focusses on a deep learning based custom-classifier to exploit the power of CNNs, that are considered to be really efficient high-dimensional feature extractors for images.

- Step 1: Data-Exploration: [DataExploration notebook](notebooks/DataExploration.html)
  - It was found during this step, that the dataset is unballenced, and I had my doubts straight away to test CNN models. I still wanted to evaluate it though, with a small CNN model described in step 3!

- Step 2: Converting the corrupted images to GrayScale: [FeatureEngineering](notebooks/FeatureEngineering.html)
    - As can also be seen from the notebook FeatureEngineering , I was able to get the most features when B-Channel of the image was given more importance, while doing the conversion.
    - The RGB image was loaded into a Numpy array, and then passed to the below formula.
  	```math #formula
		Y = 0.04 R + 0.18 G + 0.98 B

		#example: gray_image = np.dot(np_image[...,:3], [0.04, 0.18, 0.98])
	```

    - Results of the gray scale conversion is shown in below image, the last image shows how the final images will look like after feature engineering. The decision to select this was made after testing many image-processing libraries and also keeping CNN model in mind!
    
	![Results](code/screenshots/GrayScaleConversion.png) 

- Step 3: Model design, the CNN model was designed to accept single-channel gray images: [link to model selection notebook](notebooks/classification.html).
    - CNN model was modified to reach test accuray of 87%.
    - The final model architecture consists of:
      -  3 convolution layers, 
      -  with BatchNormalization and max-Pooling layers in between 
      -  followed by a flatten layer and Dense layers which maps to final number of classes(3), 
      -  Softmax classifier was then trained with Adam optimizer. 
      -  Also, detailed model  architecture is shown below.
   
   ![ModelArch](code/screenshots/modelArchitecture.png) 

- Step 4: Generating production ready code:
  - The final step of the approach was to make the work easily accesible, easy to run and provide as much flexibility to the users as possible.

## Results:
- Test Accuracy of 87% was achieved, and below is a sample plot which shows the model predictions along with the model confidence scores for each class.

  ![ModelArch](code/screenshots/PlottingPredictions.png) 

## Future Work

I have many ideas to further improve the results and approach in general. Few of them are mentioned below.
- Futher improving the dataset:
  - Collect more images if possible. 
  - If not then perform DataAugmentation in a way to really balance the dataset, specially apply it to the classes(Partially_visibility and No_visibilty). This step alone will help to increase the model accuracy. Please note* I could have done this directly on tensorflow data generator, however, this would still not mean that the data is balanced as augmentation will be applied randomly to entire dataset. The best approach to handle this would be to create a small script that augments the data in a way that the final dataset is balanced. Following are few augmentation techniques that can be applied
    - Data Rotation
    - Data Shearing
    - Random Flips
    - ...
- Improving Data preprocessing.
  - The dataset was prepared in order to keep the Blue channel as the most prominent channel.
  - As also seen from the matplotlib plotting (shown in the results section), we can think of adding binary color map to our training set!
  
- Improve the model architecture:
  - The improvements are already significant, but there are still lot of techniques that can be applied for example:
    - Adding Skip Connections between convolution layers, this will help the model to learn low leve features at deeper layers.
    - Adding residual-blocks, similar to residual networks this improves the gradient flow, and reduces the vanishing gradient problem, helping to further stabilizes the network.
    - Trying different Kernel sizes for the Convolution layers.
    - Testing the model leakyRelu as activations
    - Testing with different optimizers, note* the code already support passing RMS_prop as an option to the main.py while training.
    - The list to further improve the CNN architectures goes on and the latest research is also vast ....

- Bonus Improvement:
  - Test the approach of transfer learning, utilize the power of really extreme deep feature extractors like Inception or ResNets, pre trained on huge datasets. Adding a custom classifier on top, to make it work for the classes that we are working with!
  - The models can be easily downloaded from TFHub, tensorflow provides easy setup for this, I had this in my mind but the requirements from the project submission did not allow me to do this. The reason is that all of the pretrained models are trained with RGB images!
  - We can hack the data to support pre-trained input requirements, by simply copying the gray image to 3-channels in order to make it look like RGB to the CNNs, while keeping the data as Gray!
  
- Traditional Machine Learning Classifiers 
  - Testing traditional machine learning approaches and classifiers like SGD or Random-forest, directly provided in sklearn package. Scikit-learn also provides many ensembling approaches to classification. 
  - I did not choose this becuase I did not wanted to further do the feature engineering, for example reducing the image dimensions and converting them to a vector etc. That was also the reason behind choosing CNN,and I am pretty sure, that the approach can be further improved to reach higher accuracies.