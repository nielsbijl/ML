class PerceptronLayer:

    def __init__(self, perceptrons: list):
        self.perceptrons = perceptrons
        self.layerInput = []
        self.output = []

    def setInput(self, layerInput: list):
        if len(layerInput) >= len(self.perceptrons):
            self.layerInput = layerInput
        else:
            raise Exception("Sorry, the length of your layer input doesn't match with the length of your perceptrons!!")

    def run(self):
        self.output = []  # Reset the output
        for perceptron in self.perceptrons:
            perceptron.setInput(self.layerInput)
            perceptron.run()
            self.output.append(perceptron.output)

    def __str__(self):
        string = ""
        for perceptron in self.perceptrons:
            string += perceptron.__str__() + '\n'
        return string

