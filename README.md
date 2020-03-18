***

# Fashion-MNIST CNN Benchmarking
Fashion MNIST is a dataset which is meant to replace tradidional MNIST dataset, because it is over cited in many research papers now. It is super easy to get high accuracy results with classic MNIST-dataset. If you haven't hear about FASHION MNIST dataset before please visit the official github repository to understand the challenge --> https://github.com/zalandoresearch/fashion-mnist, before you continue reading this project. If you already are aware of the dataset, this repository will give you an idea how you can create your deepCNN models in order to achieve as high accuracy as 96%.

## Data-Preview
![Sample Fashion-MNIST dataset](screenshots/fashion-mnist-sprite.png)

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

pip install -e .
```

Now you should be ready to start a training the models provided with this repository

Sample Train command:
	
- python code/main.py --mode train --num_epochs 10


## Options for main.py
-	python code/main.py --help


## Approach

There, were many options that could have been tried but this project focusses on a deep learning based custom-classifier to exploit the power of CNNs, that are considered to be really efficient high-dimensional feature extractors for images.


## Results:
- Test Accuracy of 87% was achieved, and below is a sample plot which shows the model predictions along with the model confidence scores for each class.

| Trained Agent | Training Scores |
|---------------|-----------------|
|![Results](./Results/TrainedBananaCollectorAgent.gif) | ![Scores](./Results/BaseScores.png) |



## Future Work

