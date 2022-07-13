import numpy as np


class ShallowNN(object):
    def __init__(self, input_shape, hidden_layer, output_shape, lr, weight_init=None, hidden_act_fn=None, op_act_fn=None, optimizer=None):
        self.inputLayerSize = int(input_shape) if input_shape != '' else 5
        self.hiddenLayerSize = int(hidden_layer) if hidden_layer != '' else 3
        self.outputLayerSize = int(output_shape) if output_shape != '' else 1
        self.lr = float(lr) if lr != '' else 0.01

        self.params = self.initialize_parameters(
            self.inputLayerSize, self.hiddenLayerSize, self.outputLayerSize)

        # tests
        print(self.params)

    def initialize_parameters(self, n_x, n_h, n_y):
        """
        Argument:
        n_x -- size of the input layer
        n_h -- size of the hidden layer
        n_y -- size of the output layer

        Returns:
        params -- python dictionary containing your parameters:
                        W1 -- weight matrix of shape (n_h, n_x)
                        b1 -- bias vector of shape (n_h, 1)
                        W2 -- weight matrix of shape (n_y, n_h)
                        b2 -- bias vector of shape (n_y, 1)
        """

        # we set up a seed so that your output matches ours although the initialization is random.
        np.random.seed(2)

        W1 = np.random.randn(n_h, n_x) * 0.01
        b1 = np.zeros((n_h, 1))
        W2 = np.random.randn(n_y, n_h) * 0.01
        b2 = np.zeros((n_y, 1))

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    def forward_propagation(self, X, parameters):
        """
        Argument:
        X -- input data of size (n_x, m)
        parameters -- python dictionary containing your parameters (output of initialization function)

        Returns:
        A2 -- The sigmoid output of the second activation
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2"
        """

        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Implement Forward Propagation to calculate A2 (probabilities)

        Z1 = np.dot(W1, X) + b1
        A1 = np.tanh(Z1)
        Z2 = np.dot(W2, A1) + b2
        yhat = self.sigmoid(Z2)

        # Values needed in the backpropagation are stored in "cache". This will be given as an input to the backpropagation
        cache = {"Z1": Z1,
                 "A1": A1,
                 "Z2": Z2,
                 "A2": yhat}

        return yhat, cache

    def sigmoid(self, Z):
        return 1/(1 + np.exp(-Z))

    def relu(self, Z):
        data = [max(0, value) for value in Z]
        return np.array(data, dtype=float)

    def get_cost(self, A2, Y):
        """
        Cross-Entropy Cost Function
        """
        m = Y.shape[1]  # number of example
        # Compute the cross-entropy cost
        logprobs = logprobs = np.multiply(
            Y, np.log(A2)) + np.multiply((1-Y), np.log(1-A2))
        cost = (-1/m) * np.sum(logprobs)
        return cost

    def backward_propagation(self, parameters, cache, X, Y):
        """
        Arguments:
        parameters -- python dictionary containing our parameters 
        cache -- a dictionary containing "Z1", "A1", "Z2" and "A2".
        X -- input data of shape (2, number of examples)
        Y -- "true" labels vector of shape (1, number of examples)

        Returns:
        grads -- python dictionary containing your gradients with respect to different parameters
        """
        m = 1

        # First, retrieve W1 and W2 from the dictionary "parameters".
        W1 = parameters["W1"]
        W2 = parameters["W2"]

        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = cache["A1"]
        A2 = cache["A2"]

        # Backward propagation: calculate dW1, db1, dW2, db2.
        dZ2 = A2 - Y
        dW2 = (1/m) * np.dot(dZ2, A1.T)
        db2 = (1/m) * (np.sum(dZ2, axis=1, keepdims=True))
        dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
        dW1 = (1/m) * (np.dot(dZ1, X.T))
        db1 = (1/m) * (np.sum(dZ1, axis=1, keepdims=True))

        grads = {"dW1": dW1,
                 "db1": db1,
                 "dW2": dW2,
                 "db2": db2}

        return grads

    def update_parameters(self, parameters, grads, learning_rate):
        """
        Updates parameters using the gradient descent update

        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients 

        Returns:
        parameters -- python dictionary containing your updated parameters 
        """
        # Retrieve each parameter from the dictionary "parameters"
        W1 = parameters["W1"]
        b1 = parameters["b1"]
        W2 = parameters["W2"]
        b2 = parameters["b2"]

        # Retrieve each gradient from the dictionary "grads"
        dW1 = grads["dW1"]
        db1 = grads["db1"]
        dW2 = grads["dW2"]
        db2 = grads["db2"]

        # Update rule for each parameter
        W1 = W1 - learning_rate * dW1
        b1 = b1 - learning_rate * db1
        W2 = W2 - learning_rate * dW2
        b2 = b2 - learning_rate * db2

        parameters = {"W1": W1,
                      "b1": b1,
                      "W2": W2,
                      "b2": b2}

        return parameters

    def nn_model(self, X, Y, learning_rate, num_iterations=1):
        n_x = X.shape[0]
        n_y = Y.shape[0]
        n_h = self.hiddenLayerSize

        # Initialize parameters
        parameters = self.initialize_parameters(n_x, n_h, n_y)

        # Loop (gradient descent)
        for i in range(0, num_iterations):
            # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache"
            A2, cache = self.forward_propagation(X, parameters)
            # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost"
            cost = self.get_cost(A2, Y)
            # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads"
            grads = self.backward_propagation(parameters, cache, X, Y)
            print(grads)
            # Update rule for each parameter
            parameters = self.update_parameters(
                parameters, grads, learning_rate)
        # Returns parameters learnt by the model. They can then be used to predict output
        return parameters
