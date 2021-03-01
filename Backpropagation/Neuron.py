import math


class Neuron:
    """
    The Neuron class is a neuron node which can be used in a neuron network.
    """

    def __init__(self, weights: list, bias):
        """
        This function initializes the Neuron object
        :param weights: The weights your input will be multiplied with
        :param bias: The inverse of the threshold value, this will be added by the sum of
                        the (weights * inputs) before it goes into the activation function

        NOTE:   The weights and input needs to have the same length
                and the weights needs to be in the same order as the input
        """
        self.activationFunction = self.sigmoid
        self.input = []
        self.weights = weights
        self.bias = bias
        self.output = None

        self.error = None
        self.newWeights = []
        self.newBias = None

    def setInput(self, neuronInput: list):
        """
        This function sets the input for the neuron, this needs to match with the weights
        :param neuronInput: The input value(s) for the neuron
        """
        if len(neuronInput) == len(self.weights):
            self.input = neuronInput
        else:
            raise Exception("Sorry, the length of your input is not equal to the length of your weights!!")

    def sigmoid(self, x):
        """
        This is the sigmoid function, this will be used as the default activation function for the neuron
        :param x: The sum of all the (inputs * weights) + bias
        :return: The output of the sigmoid function
        """
        return 1 / (1 + math.exp(-x))

    def run(self):
        """
        This function will run the neuron.
        It multiplys every input with it's matching weight and takes the sum of this. It will be added by the bias.
        Then given to the activation function and this will calculate the output of this neuron.
        """
        if self.input:
            self.output = 0
            for i in range(len(self.input)):
                self.output += self.input[i] * self.weights[i]
            self.output += self.bias
            self.output = self.activationFunction(self.output)
        else:
            raise Exception("The neuron has no input, please set the input with the setInput function!")

    def setError(self, expectedOutput, weightsNextNeuron=[], errorNextNeuron=[]):
        if self.output:
            if weightsNextNeuron and errorNextNeuron:  # if this neuron is not an end neuron
                sumFromNextNodes = 0
                for i in range(len(weightsNextNeuron)):
                    sumFromNextNodes += weightsNextNeuron[i] * errorNextNeuron[i]
                self.error = self.output * (1 - self.output) * sumFromNextNodes
            else:  # if the neuron is an end neuron
                self.error = self.output * (1 - self.output) * -(expectedOutput - self.output)
        else:
            raise Exception("The neuron has no output, please run the neuron with the run function!")

    def backPropagation(self, learningRate):
        if self.error:
            self.newWeights = []
            for i in range(len(self.weights)):
                self.newWeights.append(self.weights[i] - learningRate * self.input[i] * self.error)
            self.newBias = self.bias - learningRate * self.error
        else:
            raise Exception("The neuron has no error, please set the error with the setError function!")

    def update(self):
        if self.newBias:
            if self.newWeights:
                self.weights = self.newWeights
                self.bias = self.newBias
            else:
                raise Exception("The neuron has no newWeights, please set the newWeights with the backPropagation "
                                "function!")
        else:
            raise Exception("The neuron has no newBias, please set the newBias with the backPropagation function!")

    def __str__(self):
        return f"input: {self.input}, weights: {self.weights}, bias: {self.bias}, output: {self.output}"
