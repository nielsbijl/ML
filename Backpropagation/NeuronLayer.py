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

        self.weights = [neuron.weights for neuron in self.neurons]
        self.errorNextNeurons = []
        self.errors = []

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
        return self.output

    def setErrorLayer(self, expectedOutput: list, weightsNextLayer: list = [], errorNextLayer: list = []):
        """
        This function calculates the error of this layer. It loops over all the neurons
        of this layer and sets the errors.
        :param expectedOutput: The expected output of the network for one training example
        :param weightsNextLayer: The weights to the next layer
        :param errorNextLayer: The error of the neurons of the next layer
        """
        self.errorNextNeurons = []  # Reset the errors of the neurons of the next layer
        if self.output:
            if weightsNextLayer and errorNextLayer:
                for i in range(len(self.neurons)):
                    nextWeights = []
                    for weights in weightsNextLayer:
                        nextWeights.append(weights[i])
                    self.neurons[i].setError(expectedOutput, nextWeights, errorNextLayer)
                    self.errors.append(self.neurons[i].error)
            else:
                for i in range(len(self.neurons)):
                    self.neurons[i].setError(expectedOutput[i])
                    self.errors.append(self.neurons[i].error)
        else:
            raise Exception("The neuron layer has no output, please run the neuron with the run function!")

    def backPropagationLayer(self, learningRate):
        """
        This function calculates the new weights ans biases of the neurons of this layer.
        :param learningRate: The rate you want the neurons to learn
        """
        for neuron in self.neurons:
            neuron.backPropagation(learningRate)
        self.errors = []
        self.weights = [neuron.weights for neuron in self.neurons]

    def updateLayer(self):
        """
        This function sets all the new weights and biases of the neurons in this layer.
        """
        for neuron in self.neurons:
            neuron.update()

    def __str__(self):
        string = ""
        for neuron in self.neurons:
            string += "neuron: " + neuron.__str__() + '\n'
        return string

