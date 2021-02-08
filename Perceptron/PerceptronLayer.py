class PerceptronLayer:

    def __init__(self, perceptrons: list):
        self.perceptrons = perceptrons
        self.layerInput = []
        self.output = []

    def setInput(self, layerInput: list):
        self.layerInput = layerInput

    def run(self):
        for i in range(len(self.perceptrons)):
            perceptron = self.perceptrons[i]
            perceptron.setInput(self.layerInput)
            perceptron.run()
            self.output.append(perceptron.ouput)

    def __str__(self):
        string = ""
        for perceptron in self.perceptrons:
            string += perceptron.__str__() + '\n'
        return string

