import gradient
from constants import *
from train import *
from gradient import *
import argparse
import util 
import matplotlib.pyplot as plt

# TODO
def main(args):

    # Read the required config
    # Create different config files for different experiments


    configFile = None  # Will contain the name of the config file to be loaded
    if (args.experiment == 'test_softmax'):  # Rubric #4: Softmax Regression
        configFile = "config_4.yaml"
    elif (args.experiment == 'test_gradients'):  # Rubric #5: Numerical Approximation of Gradients
        configFile = "config_5.yaml"
    elif (args.experiment == 'test_momentum'):  # Rubric #6: Momentum Experiments
        configFile = "config_6.yaml"
    elif (args.experiment == 'test_regularization'):  # Rubric #7: Regularization Experiments
        configFile = "config_7.yaml"  # Create a config file and change None to the config file name
    elif (args.experiment == 'test_activation'):  # Rubric #8: Activation Experiments
        configFile = "config_8.yaml"  # Create a config file and change None to the config file name

    print(f'Running Config "{args.experiment}" with config file "{configFile}"')

    # Load the data
    x_train, y_train, x_valid, y_valid, x_test, y_test = util.load_data(path=datasetDir)
    # Load the configuration from the corresponding yaml file. Specify the file path and name
    config = util.load_config(configYamlPath + configFile)
    nn = Neuralnetwork(config)

    
    # SOFTMAX
    if (args.experiment == 'test_softmax' or args.experiment == 'test_momentum' or args.experiment == 'test_regularization'\
         or args.experiment == 'test_activation' or args.experiment == 'test_activation'):
        trained_nn, train_acc, train_loss, val_acc, val_loss = train(nn, x_train=x_train, y_train=y_train, x_valid=x_valid, y_valid=y_valid, config=config)
        test_acc, test_loss = modelTest(trained_nn, X_test=x_test, y_test=y_test)
        util.plots(train_loss, train_acc, val_loss, val_acc)

        print('Test accuracy: ', test_acc, '; Test loss: ', test_loss)
    # GRADIENT
    if (args.experiment == 'test_gradients'):
        nn.batch_size = 1
        check_grad(nn, x_train, y_train)

if __name__ == "__main__":

    #Parse the input arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='test_momentum', help='Specify the experiment that you want to run')
    args = parser.parse_args()
    main(args)