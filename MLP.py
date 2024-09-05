import numpy as np
from abc import ABC, abstractmethod

class Layer(ABC):
    def __init__(self, input_size, output_size):
        self.input = None
        self.output = None
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.randn(input_size, output_size) * np.sqrt(2. / input_size)
        self.bias = np.zeros((1, output_size))

    @staticmethod
    def activate(z, activation: str):
        if activation == 'sigmoid':
            return 1 / (1 + np.exp(-z))
        elif activation == 'relu':
            return np.maximum(0, z)
        else:
            print("Undefined activation type")
            return None

    @staticmethod
    def derivative(activation: str, z):
        if activation == 'sigmoid':
            return z * (1 - z)
        elif activation == 'relu':
            return np.where(z > 0, 1, 0)
        else:
            print("Undefined activation type")
            return None

    @abstractmethod
    def feedforward(self, x_train):
        pass

    @abstractmethod
    def backpropagation(self, x_train, y_train, learning_rate):
        pass

class Dense(Layer):
    def __init__(self, input_size, output_size, activation: str):
        super().__init__(input_size, output_size)
        self.activation = activation

    def feedforward(self, input):
        self.input = input
        z = np.dot(self.input, self.weights) + self.bias
        self.output = self.activate(z, self.activation)
        return self.output

    def backpropagation(self, error, learning_rate):
        d = self.derivative(self.activation, self.output)
        db = np.sum(error * d, axis=0, keepdims=True)
        dW = np.dot(self.input.T, error * d)

        input_error = np.dot(error * d, self.weights.T)

        self.weights -= learning_rate * dW
        self.bias -= learning_rate * db

        return input_error

class MLP:
    def __init__(self):
        self.layers = []

    @staticmethod
    def loss_MSE(y_train, yhat):
        loss = np.mean(np.square(yhat - y_train))
        return loss

    @staticmethod
    def loss_cross(y_train, yhat):
        epsilon = 1e-9  # avoid log(0)
        loss = -np.sum(y_train * np.log(yhat + epsilon)) / y_train.shape[0]
        return loss

    def addlayer(self, layer: Dense):
        self.layers.append(layer)

    def predict(self, input):
        for layer in self.layers:
            input = layer.feedforward(input)
        return input

    def fit(self, x_train, y_train, learning_rate=0.01, batch_size=8, epochs=10, loss_type: str = 'MSE'):
        num_samples = x_train.shape[0]
        for epoch in range(epochs):
            indices = np.arange(num_samples)
            np.random.shuffle(indices)
            x_train = x_train[indices]
            y_train = y_train[indices]
            loss = 0

            for i in range(0, num_samples, batch_size):
                x_batch = x_train[i: i + batch_size]
                y_batch = y_train[i: i + batch_size]

                output = self.predict(input=x_batch)
                error = output - y_batch

                if loss_type == 'MSE':
                    loss += self.loss_MSE(y_batch, output)
                else:
                    loss += self.loss_cross(y_batch, output)

                for layer in reversed(self.layers):
                    error = layer.backpropagation(error, learning_rate)

            print(f'Epoch {epoch + 1}/{epochs}, Loss: {loss / (num_samples // batch_size)}')
