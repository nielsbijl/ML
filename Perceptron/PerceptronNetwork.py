class PerceptronNetwork:
    """
    The PerceptronNetwork class is a network of perceptrons, this contains layers which contains perceptrons
    """

    def __init__(self, layers: list):
        """
        This function initializes the PerceptronNetwork object
        :param perceptrons: The list of layer(s) for the network
        """
        self.layers = layers
        self.networkInput = []
        self.output = []

    def setInput(self, networkInput: list):
        """
        This function sets the input for the network
        :param networkInput: The list of integers or floats as input for the network
        """
        self.networkInput = networkInput

    def feedForward(self):
        """
        This functions runs the network. A layer gets his input and his output will be the next input for the next layer
        This wil generate the final output of the perceptron network
        """
        if self.networkInput:
            layerInput = self.networkInput
            for layer in self.layers:
                layer.setInput(layerInput)
                layer.run()
                layerInput = layer.output
            self.output = layerInput
        else:
            raise Exception("The perceptron network has no input, please set the input with the setInput function!")
