import argparse

def parseArguments():
    # Creates argument parser
    parser = argparse.ArgumentParser()


    # Load the model for training testing or to continue the training from a specific checkpoint!
    parser.add_argument('--mode', help='train/predict', default="train", type=str)
    
    
    # Tensorflow Graph,Session,model based parametes
    parser.add_argument('--model_arch', help='Model architecture to use custom/vgg_like/transfer_learning', default='custom')
    parser.add_argument('--save_dir', help='Path to save the trained Models', default='saved_models')
    parser.add_argument('--exp_name', help='Model will be saved with this name', default='Experiment_1')
    parser.add_argument('--load_dir', help='Path to load the trained Models', default='saved_models')
    
    # Training Parameters
    parser.add_argument('--num_classes', help='Number of Classes in fashionMNIST', type=int, default=10)
    parser.add_argument('--img_h', help='Image Height', type=int, default=28)
    parser.add_argument('--img_w', help='Image width', type=int, default=28)
    parser.add_argument('--num_channels', help='Image Channels (1 or 3) defaults to 1', type=int, default=1)
    parser.add_argument('-b', '--batch_size', help='Batch Size for training', type=int, default=250)
    parser.add_argument('--num_epochs', help='Number of Epochs to train the model for', type=int, default= 10)
    parser.add_argument('--num_aug', help='Number of image augmentation to perform', type=int, default= 2)
    
    #Select different Optimizers to test
    parser.add_argument('--optimizer', help="Optimizer to Choose RMS or Adam", default="Adam", type=str)

    

    args = parser.parse_args()
    return args