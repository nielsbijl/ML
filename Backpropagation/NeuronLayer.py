class NeuronLayer:
    """
    The NeuronLayer class is a layer for a neuron network with one or more neurons.
    """

    def __init__(self, neurons: list):
        """
        This function initializes the NeuronLayer object
        :param neurons: A list of neurons that needs to be in this layer
        """
        self.neurons = neurons
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
        This function runs every neuron in this layer
        """
        if self.layerInput:
            self.output = []  # Reset the output
            for neuron in self.neurons:
                neuron.setInput(self.layerInput)
                neuron.run()
                self.output.append(neuron.output)
        else:
            raise Exception("The neuron layer has no input, please set the input with the setInput function!")

    def __str__(self):
        string = ""
        for neuron in self.neurons:
            string += "neuron: " + neuron.__str__() + '\n'
        return string

