class Perceptron:

    def __init__(self, weights: list, bias: int or float):
        self.activationFunction = self.stepFunction
        self.input = []
        self.weights = weights
        self.bias = bias
        self.ouput = None

    def setInput(self, perceptronInput: list):
        self.input = perceptronInput

    def stepFunction(self, x):
        return 0 if x < 0 else 1

    def run(self):
        self.ouput = 0
        for i in range(len(self.input)):
            self.ouput += self.input[i] * self.weights[i]
        self.ouput -= self.bias
        self.ouput = self.activationFunction(self.ouput)

    def __str__(self):
        return f"input: {self.input}, weights: {self.weights}, bias: {self.bias}, output: {self.ouput}"

