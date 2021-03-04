class NeuronNetwork:
    """
    The NeuronNetwork class is a network of neurons, this contains layers which contains neurons
    """

    def __init__(self, layers: list):
        """
        This function initializes the NeuronNetwork object
        :param layers: The list of layer(s) for the network
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
        This wil generate the final output of the neuron network
        """
        if self.networkInput:
            layerInput = self.networkInput
            for layer in self.layers:
                layer.setInput(layerInput)
                layer.run()
                layerInput = layer.output
            self.output = layerInput
        else:
            raise Exception("The neuron network has no input, please set the input with the setInput function!")
        return self.output

    def train(self, inputs, targets, learningRate):
        for x in range(len(inputs)):
            self.setInput(inputs[x])
            self.feedForward()
            for i in range(len(self.layers)):
                i = i * -1
                if i == 0:
                    self.layers[i - 1].setErrorLayer(expectedOutput=targets[x])
                else:
                    self.layers[i - 1].setErrorLayer(expectedOutput=targets[x],
                                                     weightsNextLayer=self.layers[i].weights,
                                                     errorNextLayer=self.layers[i].errors)
            for i in range(len(self.layers)):
                i = i * -1
                self.layers[i - 1].backPropagationLayer(learningRate)
                self.layers[i - 1].updateLayer()

    def __str__(self):
        string = ''
        for i in range(len(self.layers)):
            string += f"layer: {i}: \n {self.layers[i].__str__()} \n"
        return string
