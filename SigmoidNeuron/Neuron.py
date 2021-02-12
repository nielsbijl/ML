import math


class Neuron:
    """
    The Neuron class is a perceptron node which can be used in a Neuron network.
    """

    def __init__(self, weights: list, bias: int or float):
        self.activationFunction = self.sigmoid
        self.input = []
        self.weights = weights
        self.bias = bias
        self.output = None

        self.errorSum = 0
        self.trainCount = 0
        self.MSE = None

    def setInput(self, neuronInput: list):
        if len(neuronInput) == len(self.weights):
            self.input = neuronInput
        else:
            raise Exception("Sorry, the length of your input is not equal to the length of your weights!!")

    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))

    def run(self):
        if self.input:
            self.output = 0
            for i in range(len(self.input)):
                self.output += self.input[i] * self.weights[i]
            self.output += self.bias
            self.output = self.activationFunction(self.output)
        else:
            raise Exception("The neuron has no input, please set the input with the setInput function!")

    def update(self, expectedOutput: int or float, learningRate: int or float = 0.1):
        self.run()  # Run to get an output
        output = self.output
        error = expectedOutput - output
        for i in range(len(self.weights)):
            deltaW = learningRate * error * self.input[i]
            self.weights[i] = self.weights[i] + deltaW
        deltaB = learningRate * error
        self.bias = self.bias + deltaB

        self.trainCount += 1
        self.errorSum += abs(error)

    def error(self):
        if self.trainCount:
            self.MSE = (self.errorSum ** 2) / self.trainCount
        else:
            raise Exception("The neuron isn't been trained, please train the neuron with the update function!")
        return self.MSE

    def __str__(self):
        return f"input: {self.input}, weights: {self.weights}, bias: {self.bias}, output: {self.output}"
