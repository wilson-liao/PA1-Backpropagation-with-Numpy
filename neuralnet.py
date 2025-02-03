import numpy as np
import util

class Activation():
    """
    The class implements different types of activation functions for
    your neural network layers.

    """

    def __init__(self, activation_type = "sigmoid"):
        """
        Initialize activation type and placeholders here.
        """
        if activation_type not in ["sigmoid", "tanh", "ReLU", "output"]:
            raise NotImplementedError(f"{activation_type} is not implemented.")

        # Type of non-linear activation.
        self.activation_type = activation_type

        # Placeholder for input. This can be used for computing gradients.
        self.x = None

    def __call__(self, z):
        """
        This method allows your instances to be callable.
        """
        return self.forward(z)

    def forward(self, z):
        """
        Compute the forward pass.
        """
        if self.activation_type == "sigmoid":
            return self.sigmoid(z)

        elif self.activation_type == "tanh":
            return self.tanh(z)

        elif self.activation_type == "ReLU":
            return self.ReLU(z)

        elif self.activation_type == "output":
            return self.output(z)

    def backward(self, z):
        """
        Compute the backward pass.
        """
        if self.activation_type == "sigmoid":
            return self.grad_sigmoid(z)

        elif self.activation_type == "tanh":
            return self.grad_tanh(z)

        elif self.activation_type == "ReLU":
            return self.grad_ReLU(z)

        elif self.activation_type == "output":
            return self.grad_output(z)


    def sigmoid(self, x):
        """
        TODO: Implement the sigmoid activation here.
        """
        return 1/(1 + np.exp(-x))
        #raise NotImplementedError("Sigmoid not implemented")

    def tanh(self, x):
        """
        TODO: Implement tanh here.
        """
        return np.tanh(x)
        #raise NotImplementedError("Tanh not implemented")

    def ReLU(self, x):
        """
        TODO: Implement ReLU here.
        """
        return np.maximum(x, 0)
        #raise NotImplementedError("ReLU not implemented")

    def output(self, x):
        """
        TODO: Implement softmax function here.
        Remember to take care of the overflow condition (i.e. how to avoid denominator becoming zero).
        """
        x = np.array(x)  # Convert input list to numpy array
        exp_values = np.exp(x - np.max(x, axis=1, keepdims=True))  # Prevent overflow
        softmax_output = exp_values / np.sum(exp_values, axis=1, keepdims=True)  # Normalize

        return softmax_output
        #raise NotImplementedError("output activation not implemented")

    def grad_sigmoid(self, x):
        """
        TODO: Compute the gradient for sigmoid here.
        """
        return self.sigmoid(x)*(1-self.sigmoid(x))
        #raise NotImplementedError("Sigmoid gradient not implemented")

    def grad_tanh(self, x):
        """
        TODO: Compute the gradient for tanh here.
        """
        return 1 - self.tanh(x)**2
        #raise NotImplementedError("Tanh gradient not implemented")

    def grad_ReLU(self, x):
        """
        TODO: Compute the gradient for ReLU here.
        """
        return np.where(x>0, 1, 0)
        #raise NotImplementedError("ReLU gradient not implemented")

    def grad_output(self, x):
        """
        Deliberately returning 1 for output layer case since we don't multiply by any activation for final layer's delta. Feel free to use/disregard it
        """
        return 1  #Deliberately returning 1 for output layer case


class Layer():
    """
    This class implements Fully Connected layers for your neural network.
    """

    def __init__(self, in_units, out_units, activation):
        """
        Define the architecture and create placeholders.
        """
        np.random.seed(42)

        # Randomly initialize weights
        self.w = 0.01 * np.random.random((in_units + 1, out_units))
        self.v = 0
        self.x = None    # Save the input to forward in this
        self.a = None    # output without activation
        self.z = None    # Output After Activation
        self.activation=activation


        self.dw = 0  # Save the gradient w.r.t w in this. w already includes bias term

    def __call__(self, x):
        """
        Make layer callable.
        """
        return self.forward(x)

    def forward(self, x):
        """
        TODO: Compute the forward pass (activation of the weighted input) through the layer here and return it.
        """
        # bias term is the last column of w
        ##############
        self.x = util.append_bias(x)
        self.a = np.dot(self.x, self.w)
        self.z = self.activation(self.a)
        return self.z
        #raise NotImplementedError("Forward propagation not implemented for Layer")

    def backward(self, deltaCur, batch_size, learning_rate, momentum_gamma, regularization, gradReqd=True):
        """
        TODO
        Write the code for backward pass. This takes in gradient from its next layer as input and
        computes gradient for its weights and the delta to pass to its previous layers. gradReqd is used to specify whether to update 
        the weights i.e. whether self.w should be updated after calculating self.dw

        The delta expression for any layer consists of delta and weights from the next layer and derivative of the activation function
        of weighted inputs i.e. g'(a) of that layer. Hence deltaCur (the input parameter) will have to be multiplied with the 
        derivative of the activation function of the weighted
        input of the current layer to actually get the delta for the current layer. Remember, this is just one way of interpreting it 
        and you are free to interpret it any other way.
        Feel free to change the function signature if you think of an alternative way to implement the delta calculation or the backward pass

        When implementing softmax regression part, just focus on implementing the single-layer case first.
        """
        # before obtaining delta we should get a matrix that's 128x785x10
        # then we average over the first (128) dimension
        

        delta = deltaCur*self.activation.backward(self.a)
        self.x = np.array(self.x)
        self.dw = momentum_gamma * self.dw - np.dot(self.x.T, delta) * (1/batch_size)

        if gradReqd:
            #L1 Update
            # self.w -= learning_rate * (self.dw - regularization * np.sign(self.w))

            #L2 Update
            self.w -= learning_rate * (self.dw - regularization * self.w)

        delta_prev = np.dot(deltaCur, self.w[:-1, :].T)
        return delta_prev
        #raise NotImplementedError("Backwards propagation not implemented for Layer")


class Neuralnetwork():
    """
    Create a Neural Network specified by the network configuration mentioned in the config yaml file.
    """

    def __init__(self, config):
        """
        Create the Neural Network using config.
        """
        self.layers = []  # Store all layers in this list.
        self.num_layers = len(config['layer_specs']) - 1  # Set num layers here
        self.x = None  # Save the input to forward in this
        self.y = None  # For saving the output vector of the model
        self.targets = None  # For saving the targets
        self.batch_size = config['batch_size']
        self.early_stop = config['early_stop']
        self.learning_rate = config['learning_rate']
        self.momentum_gamma = config['momentum_gamma']
        self.L2_pentalty = config['L2_penalty']
        self.momentum = config['momentum']
        if not self.momentum:
            self.momentum_gamma = 0

        # Add layers specified by layer_specs.
        for i in range(self.num_layers):
            if i < self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i+1], Activation(config['activation'])))
            elif i  == self.num_layers - 1:
                self.layers.append(Layer(config['layer_specs'][i], config['layer_specs'][i + 1], Activation("output")))

    def __call__(self, x, targets=None):
        """
        Make NeuralNetwork callable.
        """
        return self.forward(x, targets)

    def forward(self, x, targets=None):

        """
        TODO
        Compute forward pass through all the layers in the network and return the loss.
        If targets are provided, return loss and accuracy/number of correct predictions as well.
        
        Args:
            x: Input data.
            targets: Target labels (if provided).
        
        Returns:
            If targets are provided, returns accuracy and loss.
            If targets are not provided, returns the computed output.
        """
        self.x = x
        # y = []
        self.targets = targets
        
        currRep = x
        for layer in self.layers:
            currRep = layer.forward(currRep)
        self.y = currRep

        
        if targets is not None:
            # Predicted class is the index of the maximum logit value along axis 1
            predictions = np.argmax(self.y, axis=1)
            # True class is the index of the 1 in the one-hot encoded targets
            true_labels = np.argmax(targets, axis=1)
            # Calculate accuracy as the mean of correctly predicted labels
            accuracy = np.sum(predictions == true_labels)/len(predictions)
            loss = self.loss(self.y, targets)
            return accuracy, loss
        else:
            return currRep
        #raise NotImplementedError("Forward propagation not implemented for NeuralNetwork")


    def loss(self, logits, targets):

        '''
        TODO
        Compute the categorical cross-entropy loss and return it.
        
        Args:
            logits: The predicted logits or probabilities.
            targets: The true target labels.
        
        Returns:
            The categorical cross-entropy loss.
        '''

        return -np.mean(np.sum(targets * np.log(logits+1e-10), axis=1))    
        #raise NotImplementedError("loss not implemented for NeuralNetwork")

    def backward(self, gradReqd=True):
        
        '''
        TODO
        Implement backpropagation here by calling the backward method of Layers class.
        Call backward methods of individual layers.
        
        Args:
            gradReqd: A boolean flag indicating whether to update the weights.
        '''

        deltaCurr = (np.array(self.targets)-np.array(self.y))
        
        for layer in self.layers[::-1]:
            deltaCurr = layer.backward(deltaCurr, self.batch_size, self.learning_rate, self.momentum_gamma, self.L2_pentalty,gradReqd)
        #raise NotImplementedError("Backwards propagation not implemented for NeuralNetwork")
