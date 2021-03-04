import unittest
from Backpropagation.Neuron import *
from Backpropagation.NeuronLayer import *
from Backpropagation.NeuronNetwork import *


class testNeuronNetwork(unittest.TestCase):
    def testXORgate(self):

        # De waarheidstabel van de XOR gate
        inputs = [[1, 1],
                  [1, 0],
                  [0, 1],
                  [0, 0]]
        targets = [[0], [1], [1], [0]]

        # Het netwerk maken
        f = Neuron(weights=[0.2, -0.4], bias=0)
        g = Neuron(weights=[0.7, 0.1], bias=0)
        firstLayer = NeuronLayer(neurons=[f, g])
        o = Neuron(weights=[0.6, 0.9], bias=0)
        secondLayer = NeuronLayer(neurons=[o])
        xorNetwork = NeuronNetwork(layers=[firstLayer, secondLayer])

        # Checken dat het netwerk nu nog niet werkt
        for i in range(len(inputs)):
            xorNetwork.setInput(inputs[i])
            self.assertNotAlmostEqual(targets[i][0], xorNetwork.feedForward()[0], delta=0.1)

        # Het netwerk trainen
        for epoch in range(1000):
            xorNetwork.train(inputs, targets, 1)

        # Laten zien dat het netwerk nu werkt
        for i in range(len(inputs)):
            xorNetwork.setInput(inputs[i])
            self.assertAlmostEqual(targets[i][0], xorNetwork.feedForward()[0], delta=0.1)


if __name__ == '__main__':
    unittest.main()
