class PerceptronLayer:
    """
    The PerceptronLayer class is a layer for a perceptron network with one or more perceptrons.
    """

    def __init__(self, perceptrons: list):
        """
        This function initializes the PerceptronLayer object
        :param perceptrons: A list of perceptrons that needs to be in this layer
        """
        self.perceptrons = perceptrons
        self.layerInput = []
        self.output = []

    def setInput(self, layerInput: list):
        """
        This function sets the input of this layer
        :param layerInput: The input for this layer
        """
        self.layerInput = layerInput

    def run(self):
        """
        This function runs every perceptron in this layer
        """
        if self.layerInput:
            self.output = []  # Reset the output
            for perceptron in self.perceptrons:
                perceptron.setInput(self.layerInput)
                perceptron.run()
                self.output.append(perceptron.output)
        else:
            raise Exception("The perceptron layer has no input, please set the input with the setInput function!")

    def __str__(self):
        string = ""
        for perceptron in self.perceptrons:
            string += perceptron.__str__() + '\n'
        return string

