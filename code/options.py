import argparse

def parseArguments():
    # Creates argument parser
    parser = argparse.ArgumentParser()


    # Load the model for training testing or to continue the training from a specific checkpoint!
    parser.add_argument('--mode', help='train/predict', default="train", type=str)
    
    # Path to the dataset Directory. It should have the below Folder Structure, if not have a look at preprocessing.py file provided in util folder

    ################# Folder Structure:###########
    #    DIR:
    #       |-- images
    #           |-- img1.png
    #           |-- img2.png
    #           |-- img3.png
    #           |-- img4.png
    #           |-- img5.png
    #           |-- img6.png
    #           |-- .......
    #           |-- .......
    #       
    #       |-- .csv file that holds image names and classes!             
    ###############################################

    parser.add_argument('--input_dir', help='Path to the Dataset directory(DIR)', default='data', type=str)
    parser.add_argument('--images_dir', help='Path to the Images', default='GrayImages', type=str)
    parser.add_argument('--csv_filename', help="Name of the csv, which holds main data", default="gicsd_labels.csv", type=str)
    parser.add_argument('--output_dir', help='Path to save the preprocesed Dataset directory(DIR)', default='data', type=str)


    # Tensorflow Graph,Session,model based parametes
    parser.add_argument('--save_dir', help='Path to save the trained Models', default='artifacts')
    parser.add_argument('--exp_name', help='Model will be saved with this name', default='Experiment_1')
    parser.add_argument('--load_dir', help='Path to load the trained Models', default='artifacts')
    
    # Training Parameters
    parser.add_argument('--img_h', help='Image Height', type=int, default=192)
    parser.add_argument('--img_w', help='Image width', type=int, default=192)
    parser.add_argument('--num_channels', help='Image Channels (1 or 3) defaults to 1', type=int, default=1)
    parser.add_argument('-b', '--batch_size', help='Batch Size for training', type=int, default=16)
    parser.add_argument('--num_epochs', help='Number of Epochs to train the model for', type=int, default= 15)
    parser.add_argument('--optimizer', help="Optimizer to Choose RMS or Adam", default="Adam", type=str)

    # Prediction Parameters
    parser.add_argument('--predict_mode', help='Prediction mode single/batch, Whether to do prediction on single image or entire directory', default='single')
    parser.add_argument('--predict_image', help='path to the image to make prediction on', default='data/GrayImages')

    args = parser.parse_args()
    return args