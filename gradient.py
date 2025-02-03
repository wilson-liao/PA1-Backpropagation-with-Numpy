import numpy as np
from neuralnet import Neuralnetwork
from copy import deepcopy

def check_grad(model, x_train, y_train, epsilon=1e-2):

    """
    TODO
    Checks if gradients computed numerically are within O(epsilon**2)

    Args:
        model: The neural network model to check gradients for.
        x_train: Small subset of the original train dataset.
        y_train: Corresponding target labels of x_train.
        epsilon: Small constant for numerical approximation.

    Prints gradient difference of values calculated via numerical approximation and backprop implementation.
    """
    x_input = np.array(x_train[0]).reshape(1, -1)
    y_input = np.array(y_train[0]).reshape(1, -1)

    temp_model = deepcopy(model)
    temp_model2 = deepcopy(model)
    _, _ = temp_model2.forward(x_input, y_input)
    temp_model2.backward(False)

    for layer in range(2):
        for row in range(3):
            temp_model = deepcopy(model)
            print("checking weight at layer ", layer, " index [", row, ", 0]")
              # Ensure it's a 2D array
              # Ensure y_train is also 2D if needed
            print(f'temp_model2.layers[{layer}].dw[{row}, 0]: {temp_model2.layers[layer].dw[row, 0]}')
            temp_model.layers[layer].w[row, 0] += epsilon
            _, plus_loss = temp_model.forward(x_input, y_input)
            temp_model.layers[layer].w[row, 0] -= 2*epsilon
            _, minus_loss = temp_model.forward(x_input, y_input)
            approx = (plus_loss-minus_loss)/(2*epsilon)
            backprop = temp_model2.layers[layer].dw[row, 0]
            print("approximation value: ", approx, " backprop value: ", backprop, " difference: ", approx-backprop)
    # raise NotImplementedError("check_grad not implemented in gradient.py")



def checkGradient(model, x_train, y_train, epsilon):
    raise NotImplementedError("checkGradient not implemented in gradient.py")
