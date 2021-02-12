import unittest
from SigmoidNeuron.NeuronNetwork import *
from SigmoidNeuron.Neuron import *
from SigmoidNeuron.NeuronLayer import *


class TestNeuronNetwork(unittest.TestCase):

    def testNeuronNetworkHalfAdder(self):
        """
        This functions tests every possible input of the half adder neuron network.
        """

        """ -----Create half adder network----- """
        """ Create First hidden layer """
        orNeuron = Neuron(weights=[100, 100], bias=-50)
        nandNeuron = Neuron(weights=[-80, -80], bias=100)
        andNeuron = Neuron(weights=[80, 80], bias=-100)
        firstLayer = NeuronLayer(neurons=[orNeuron, nandNeuron, andNeuron])

        """ Create second hidden layer """
        andNeuron = Neuron(weights=[50, 50, -10], bias=-80)
        extraNeuron = Neuron(weights=[-100, -100, 1000], bias=0)
        secondLayer = NeuronLayer(neurons=[andNeuron, extraNeuron])

        """ Create neuron network """
        halfAdderNetwork = NeuronNetwork(layers=[firstLayer, secondLayer])

        """ ------Test every possible input----- """
        inputNetwork = [[[1, 1], [0, 1]],  # [[input], [output]]
                        [[1, 0], [1, 0]],
                        [[0, 1], [1, 0]],
                        [[0, 0], [0, 0]]]

        for testInput in inputNetwork:
            """ Set the input for the perceptron network """
            halfAdderNetwork.setInput(networkInput=testInput[0])

            """ Run perceptron network """
            halfAdderNetwork.feedForward()

            """ Test of the output is correct """
            output = [round(halfAdderNetwork.output[0]), round(halfAdderNetwork.output[1])]
            self.assertEqual(output, testInput[1])

            """
            Alleen door de float output af te ronden naar een integer is het mogelijk om met neurons een half adder
            te maken!
            """