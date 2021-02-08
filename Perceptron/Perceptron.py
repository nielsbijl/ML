class Perceptron:

    def __init__(self, weights: list, bias: int or float):
        self.activationFunction = self.stepFunction
        self.input = []
        self.weights = weights
        self.bias = bias
        self.output = None

    def setInput(self, perceptronInput: list):
        if len(perceptronInput) == len(self.weights):
            self.input = perceptronInput
        else:
            raise Exception("Sorry, the length of your input is not equal to the length of your weights!!")

    def stepFunction(self, x):
        return 0 if x < 0 else 1

    def run(self):
        self.output = 0
        for i in range(len(self.input)):
            self.output += self.input[i] * self.weights[i]
        self.output -= self.bias
        self.output = self.activationFunction(self.output)

    def __str__(self):
        return f"input: {self.input}, weights: {self.weights}, bias: {self.bias}, output: {self.output}"

