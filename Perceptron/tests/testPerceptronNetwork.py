import unittest
from Perceptron.PerceptronNetwork import *
from Perceptron.Perceptron import *
from Perceptron.PerceptronLayer import *


class TestPerceptronNetwork(unittest.TestCase):


    def testPerceptronNetwork(self):
        inputNetwork = [[[1, 1], 0],
                        [[1, 0], 1],
                        [[0, 1], 1],
                        [[0, 0], 0]]

        """ First hidden layer """
        orPerceptron = Perceptron(weights=[1, 1], bias=1)
        nandPerceptron = Perceptron(weights=[-1, -1], bias=-1)
        firstLayer = PerceptronLayer(perceptrons=[orPerceptron, nandPerceptron])

        """ Second hidden layer """
        andPerceptron = Perceptron(weights=[1, 1], bias=2)
        secondLayer = PerceptronLayer(perceptrons=[andPerceptron])

        """ Perceptron network """
        xorNetwork = PerceptronNetwork(layers=[firstLayer, secondLayer])

        """ Set the input for the perceptron network """
        xorNetwork.setInput(networkInput=[0, 1])

        """ Run perceptron network """
        xorNetwork.feedForward()

        """ Test of the output is correct """
        self.assertEqual(xorNetwork.output[0], 1)

        xorNetwork.setInput(networkInput=[0, 0])

        """ Run perceptron network """
        xorNetwork.feedForward()

        """ Test of the output is correct """
        self.assertEqual(xorNetwork.output[0], 0)