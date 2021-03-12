import time
import math


def normalize2DList(inputs: list):
    """
    This function normalizes a 2 dimensional list
    :param inputs: 2d list of inputs
    :return: 2d list of normalized inputs
    """
    normalizedInputs = []
    maxInput = 1
    for row in inputs:  # Searching the max of the inputs
        rowMax = max(row)
        if rowMax > maxInput:
            maxInput = rowMax
    for row in inputs:
        normalizedRow = []
        for item in row:
            normalizedRow.append(item / maxInput)
        normalizedInputs.append(normalizedRow)
    return normalizedInputs


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

        self.losses = []
        self.MSE = None

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

    def calculateLoss(self, expectedOutput: list):
        """
        This function calculates the loss of the network with one training example
        :param expectedOutput:
        :return: The loss of one training example
        """
        lossSum = 0
        for i in range(len(expectedOutput)):
            lossSum += (expectedOutput[i] - self.output[i]) ** 2
        self.losses.append(lossSum)
        self.MSE = None  # Reset the MSE
        return lossSum

    def calculateTotalLoss(self, inputs: list, targets: list):
        """
        This function calculates the total loss of the network (AKA MSE)
        :return: Mean Squared Error of the neural network
        """
        for x in range(len(inputs)):  # Alle losses uitrekenen over het netwerk
            self.setInput(inputs[x])
            self.feedForward()
            self.calculateLoss(expectedOutput=targets[x])

        MSE = sum(self.losses) / len(self.losses)
        self.losses = []  # Reset the losses
        self.MSE = MSE
        return MSE

    def train(self, inputs: list, targets: list, learningRate):
        """
        This function will train the neural network for 1 epoch
        :param inputs: A 2d array of inputs
        :param targets: A 2d array of targets
        :param learningRate: The rate you want the network to train
        """
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
        self.calculateTotalLoss(inputs, targets)

    def trainBatches(self, inputs: list, targets: list, learningRate, batchSize: int):
        # Note: Ik heb de normale train functie ook laten staan zodat het overzichtelijk wordt wat ik doe.
        #       Het is natuurlijk ook mogelijk om deze functie een batchSize van 1 te geven,
        #       dan doet die precies het zelfde als de normale train functie.
        """
        This function trains this network for 1 epoch
        :param batchSize: The size of the batch it will calculates his mean error on
        :param inputs: A 2d array of inputs
        :param targets: A 2d array of targets
        :param learningRate: The rate you want the network to train
        """
        index0, indexBatchSize = 0, batchSize
        batchInput, batchTarget = inputs[index0: indexBatchSize], targets[index0: indexBatchSize]
        for y in range(math.ceil(len(inputs) / batchSize)):
            outputErrors = [[] for temp in range(len(self.layers[-1].neurons))]  # Create list of empty lists
            for x in range(len(batchInput)):
                self.setInput(batchInput[x])
                self.feedForward()
                self.layers[-1].setErrorLayer(expectedOutput=batchTarget[x])  # Calculate the output errors
                for z in range(len(self.layers[-1].errors)):
                    outputErrors[z].append(self.layers[-1].errors[z])
                self.layers[-1].errors = []
            index0, indexBatchSize = index0 + batchSize, indexBatchSize + batchSize
            batchInput, batchTarget = inputs[index0: indexBatchSize], targets[index0: indexBatchSize]
            for errorIndex in range(len(self.layers[-1].neurons)):
                outputErrors[errorIndex] = sum(outputErrors[errorIndex]) / len(outputErrors[errorIndex])  # Calculate the mean of the output errors of the batch
            self.layers[-1].errors = outputErrors
            for i in range(len(self.layers)):
                i = i * -1
                if i == 0:
                    for neuronIndex in range(len(self.layers[i - 1].neurons)):  # Set the mean errors to the output neurons
                        self.layers[i - 1].neurons[neuronIndex].error = outputErrors[neuronIndex]
                else:
                    self.layers[i - 1].setErrorLayer(expectedOutput=targets[x],  # Calculate the rest of the errors of the network
                                                     weightsNextLayer=self.layers[i].weights,
                                                     errorNextLayer=self.layers[i].errors)
            for i in range(len(self.layers)):
                i = i * -1
                self.layers[i - 1].backPropagationLayer(learningRate)
                self.layers[i - 1].updateLayer()
        self.calculateTotalLoss(inputs, targets)  # Calculate the MSE of the current network

    def fit(self, inputs: list, targets: list, learningRate, epochs: int = 1, batchSize: int = 1, maxMSE=0,
            maxTime: int = None):
        """
        This function fits the neural network
        :param batchSize:
        :param inputs: A 2d array of inputs
        :param targets: A 2d array of outputs
        :param learningRate: The rate you want the network to learn
        :param epochs: The maximal epochs the network will train with
        :param maxMSE: If the network has this MSE it will stop training
        :param normalizeInputs: If you want the network to uses the inputs normalized
        :param maxTime: The maximal of seconds the network is allowed to train
        """
        startTime = time.time()
        for epoch in range(epochs):
            if batchSize > 1:
                self.trainBatches(inputs=inputs, targets=targets, learningRate=learningRate, batchSize=batchSize)
            else:
                self.train(inputs=inputs, targets=targets, learningRate=learningRate)
            if self.MSE < maxMSE:
                break
            if maxTime:
                if (time.time() - startTime) > maxTime:
                    break

    def score(self, inputs: list, targets: list):
        """
        This function calculates the score of the neural network on the given data
        :param inputs: 2 dimensional list of inputs
        :param targets: 2 dimensional list of targets
        :return: The score of the neural network
        """
        true = 0
        for i in range(len(inputs)):
            self.setInput(inputs[i])
            outp = self.feedForward()
            if outp.index(max(outp)) == targets[i].index(max(targets[i])):
                true += 1
        return true / len(targets) * 100

    def __str__(self):
        string = ''
        for i in range(len(self.layers)):
            string += f"layer: {i}: \n {self.layers[i].__str__()} \n"
        return string
