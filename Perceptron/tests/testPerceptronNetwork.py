import unittest
from Perceptron.PerceptronNetwork import *
from Perceptron.Perceptron import *
from Perceptron.PerceptronLayer import *


class TestPerceptronNetwork(unittest.TestCase):
    def testPerceptronNetworkXOR(self):
        """
        This functions tests every possible input of the XOR perceptron network.
        """

        """ -----Create XOR network----- """

        """ Create First hidden layer """
        orPerceptron = Perceptron(weights=[1, 1], bias=1)
        nandPerceptron = Perceptron(weights=[-1, -1], bias=-1)
        firstLayer = PerceptronLayer(perceptrons=[orPerceptron, nandPerceptron])

        """ Create second hidden layer """
        andPerceptron = Perceptron(weights=[1, 1], bias=2)
        secondLayer = PerceptronLayer(perceptrons=[andPerceptron])

        """ Create perceptron network """
        self.xorNetwork = PerceptronNetwork(layers=[firstLayer, secondLayer])

        """ ------Test every possible input----- """
        inputNetwork = [[[1, 1], 0],
                        [[1, 0], 1],
                        [[0, 1], 1],
                        [[0, 0], 0]]

        for testInput in inputNetwork:
            """ Set the input for the perceptron network """
            self.xorNetwork.setInput(networkInput=testInput[0])

            """ Run perceptron network """
            self.xorNetwork.feedForward()

            """ Test of the output is correct """
            self.assertEqual(self.xorNetwork.output[0], testInput[1])

    def testPerceptronNetworkHalfAdder(self):
        """
        This functions tests every possible input of the half adder perceptron network.
        """

        """ -----Create half adder network----- """

        """ Create First hidden layer """
        orPerceptron = Perceptron(weights=[1, 1], bias=1)
        nandPerceptron = Perceptron(weights=[-1, -1], bias=-1)
        andPerceptron = Perceptron(weights=[1, 1], bias=2)
        firstLayer = PerceptronLayer(perceptrons=[orPerceptron, nandPerceptron, andPerceptron])

        """ Create second hidden layer """
        andPerceptron = Perceptron(weights=[1, 1, 0], bias=2)
        extraPerceptron = Perceptron(weights=[0, 0, 1], bias=1)
        secondLayer = PerceptronLayer(perceptrons=[andPerceptron, extraPerceptron])

        """ Create perceptron network """
        self.halfAdderNetwork = PerceptronNetwork(layers=[firstLayer, secondLayer])

        """ ------Test every possible input----- """
        inputNetwork = [[[1, 1], [0, 1]],  # [[input], [output]]
                        [[1, 0], [1, 0]],
                        [[0, 1], [1, 0]],
                        [[0, 0], [0, 0]]]
        for testInput in inputNetwork:
            """ Set the input for the perceptron network """
            self.halfAdderNetwork.setInput(networkInput=testInput[0])

            """ Run perceptron network """
            self.halfAdderNetwork.feedForward()

            """ Test of the output is correct """
            self.assertEqual(self.halfAdderNetwork.output, testInput[1])
