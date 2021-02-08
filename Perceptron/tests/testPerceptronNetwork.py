import unittest
from Perceptron.PerceptronNetwork import *
from Perceptron.Perceptron import *
from Perceptron.PerceptronLayer import *


class TestPerceptronNetwork(unittest.TestCase):
    def setUp(self):
        """
        This function gets executed before every test function gets executed
        It creates a XOR perceptron network
        """

        """ Create First hidden layer """
        orPerceptron = Perceptron(weights=[1, 1], bias=1)
        nandPerceptron = Perceptron(weights=[-1, -1], bias=-1)
        firstLayer = PerceptronLayer(perceptrons=[orPerceptron, nandPerceptron])

        """ Create second hidden layer """
        andPerceptron = Perceptron(weights=[1, 1], bias=2)
        secondLayer = PerceptronLayer(perceptrons=[andPerceptron])

        """ Create perceptron network """
        self.xorNetwork = PerceptronNetwork(layers=[firstLayer, secondLayer])

    def testPerceptronNetwork(self):
        """
        This functions tests every possible input of the XOR perceptron network.
        """

        inputNetwork = [[[1, 1], 0],
                        [[1, 0], 1],
                        [[0, 1], 1],
                        [[0, 0], 0]]
        """ Test every possible input """
        for testInput in inputNetwork:
            """ Set the input for the perceptron network """
            self.xorNetwork.setInput(networkInput=testInput[0])

            """ Run perceptron network """
            self.xorNetwork.feedForward()

            """ Test of the output is correct """
            self.assertEqual(self.xorNetwork.output[0], testInput[1])



