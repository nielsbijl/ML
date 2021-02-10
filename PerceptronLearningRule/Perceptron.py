import math


class Perceptron:
    """
    The Perceptron class is perceptron node which can be used in a perceptron network.
    """

    def __init__(self, weights: list, bias: int or float):
        """
        This function initializes the Perceptron object
        :param weights: The weights your input will be multiplied with
        :param bias: The threshold value before a perceptron will by activated

        NOTE:   The weights and input needs to have the same length
                and the weights needs to be in the same order as the input
        """
        self.activationFunction = self.stepFunction
        self.input = []
        self.weights = weights
        self.bias = bias
        self.output = None

    def setInput(self, perceptronInput: list):
        """
        This function sets the input for the perceptron, this needs to match with the weights
        :param perceptronInput: The input value(s) for the perceptron
        """
        if len(perceptronInput) == len(self.weights):
            self.input = perceptronInput
        else:
            raise Exception("Sorry, the length of your input is not equal to the length of your weights!!")

    def stepFunction(self, x):
        """
        This is the step function, this will be used as the default activation function for the perceptron
        :param x: The sum of all the (inputs * weights) - bias
        :return: If the perceptron will be active or not
        """
        return 0 if x < 0 else 1

    # def sigmoid(self, x):
    #     return 1 / (1 + math.exp(-x))

    def run(self):
        """
        This function will run the perceptron.
        It multiplys every input with it's matching weight and takes the sum of this. It will be subtracted by the bias.
        Then given to the activation function and this will chose of the perceptron is active or not.
        :return:
        """
        if self.input:
            self.output = 0
            for i in range(len(self.input)):
                self.output += self.input[i] * self.weights[i]
            self.output += self.bias
            self.output = self.activationFunction(self.output)
        else:
            raise Exception("The perceptron has no input, please set the input with the setInput function!")

    def update(self, expectedOutput: int or float, learningRate: int or float = 0.1):
        """
        This function contains the Perceptron Learning Rule
        The Perceptron Learning Rule makes it possible to fit the perceptron

        :param expectedOutput: The expected output from this perceptron
        :param learningRate: η The learning rate of this perceptron, default value = 0.1
        """
        self.run()  # Run to get an output
        output = self.output
        error = expectedOutput - output
        for i in range(len(self.weights)):
            deltaW = learningRate * error * self.input[i]
            self.weights[i] = self.weights[i] + deltaW
        deltaB = learningRate * error
        self.bias = self.bias + deltaB

    def __str__(self):
        return f"input: {self.input}, weights: {self.weights}, bias: {self.bias}, output: {self.output}"