***

# Fashion-MNIST CNN Benchmarking


## Data


## Dependencies:

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

pip install -r requirements.txt
```

Now you should be ready to start a training or predict the trained model on any image(RGB/Gray) both are automatically handeled by the code/predict.py code!	

Sample Train command:
	
- python code/main.py --mode train


## Options for main.py
	--load_dir = directory where the model is saved after training

	--exp_name = name of the experiment that was mentioned during training defaults to Experiment_1. This is also taken as the name of the model while saving it.
	
	--predict_image = Path to the image #This can be both "either RGB or Gray Scale image". Predict.py makes sure that the proper preprocessing and inference pipelines are run to this image before getting the model predictions.

	The project was coded to provide maximum flexibilty, therefore, you can check more parameters that can be tweeked in code/options.py or simply run 
	
	"python code/main.py --help"


## Approach

There, were many options that could have been tried but this project focusses on a deep learning based custom-classifier to exploit the power of CNNs, that are considered to be really efficient high-dimensional feature extractors for images.


## Results:
- Test Accuracy of 87% was achieved, and below is a sample plot which shows the model predictions along with the model confidence scores for each class.

| Trained Agent | Training Scores |
|---------------|-----------------|
|![Results](./Results/TrainedBananaCollectorAgent.gif) | ![Scores](./Results/BaseScores.png) |



## Future Work

