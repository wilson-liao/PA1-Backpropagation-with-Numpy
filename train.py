import copy
from neuralnet import *

def train(model, x_train, y_train, x_valid, y_valid, config):
    """
    TODO: Train your model here.
    Implements mini-batch SGD to train the model.
    Implements Early Stopping.
    Uses config to set parameters for training like learning rate, momentum, etc.

    args:
        model - an object of the NeuralNetwork class
        x_train - the train set examples
        y_train - the test set targets/labels
        x_valid - the validation set examples
        y_valid - the validation set targets/labels

    returns:
        the trained model
    """
    epochs = config['epochs']
    batch_size = config['batch_size']
    early_stop_epoch = config['early_stop_epoch']
    batches = int(len(x_train)/batch_size)
    train_losses = []
    train_accuracy = []
    val_losses = []
    val_accuracy = []

    for i in range(epochs): # epochs methinks lmao idk
        #randomize order of training set
        randList = np.arange(len(x_train))
        np.random.shuffle(randList)
        x_train = [x_train[i] for i in randList]
        y_train = [y_train[i] for i in randList]
        temploss = []
        tempacc = []
        for j in range(batches):
            start = j*batch_size
            end = (j+1)*batch_size
            #do a forward pass to figure out current prediction
            accuracy, loss = model.forward(x_train[start:end], y_train[start:end])
            # print(accuracy, loss)
            temploss.append(loss)
            tempacc.append(accuracy)
            #do a backward pass to update w
            model.backward(True)
        train_losses.append(sum(temploss)/len(temploss))
        train_accuracy.append(sum(tempacc)/len(tempacc))
        print("Epoch ",i, train_accuracy[i], train_losses[i])
        accuracy, loss = model.forward(x_valid, y_valid)
        val_losses.append(loss)
        val_accuracy.append(accuracy)
        
        #early stop
        if i > early_stop_epoch and val_accuracy[i] <= val_accuracy[i-early_stop_epoch] and val_losses[i] >= val_losses[i-early_stop_epoch]:
            print(f'Early Stopped at epoch {i}')
            print(accuracy, loss, val_accuracy, val_losses, train_accuracy, train_losses)
            return model, train_accuracy, train_losses, val_accuracy, val_losses
    print(accuracy, loss, val_accuracy, val_losses, train_accuracy, train_losses)
    return model, train_accuracy, train_losses, val_accuracy, val_losses
    
    #raise NotImplementedError("Train function not implemented")

def modelTest(model, X_test, y_test):
    """
    TODO
    Calculates and returns the accuracy & loss on the test set.

    args:
        model - the trained model, an object of the NeuralNetwork class
        X_test - the test set examples
        y_test - the test set targets/labels

    returns:
        test accuracy
        test loss
    """
    accuracy, loss = model.forward(X_test, y_test)
    return accuracy, loss
    #raise NotImplementedError("Test function not implemented")