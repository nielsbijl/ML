import unittest
from SigmoidNeuron.Neuron import *


class TestNeuron(unittest.TestCase):
    """2.a"""
    def testNeuronANDGATE(self):
        """ Testing of the neuron classe with the AND gate with every possible input"""
        andNeuron = Neuron(weights=[1, 1], bias=-2)

        inputAndNeuron = [[[1, 1], 1],
                              [[1, 0], 0],
                              [[0, 1], 0],
                              [[0, 0], 0]]

        for testInput in inputAndNeuron:
            andNeuron.setInput(neuronInput=testInput[0])
            andNeuron.run()
            self.assertNotEqual(andNeuron.output, testInput[1])

        """
        De andNeuron gate is alleen maar mogelijk bij het afronden van de output floats naar integers 
        """
        andNeuron = Neuron(weights=[80, 80], bias=-100)

        for testInput in inputAndNeuron:
            andNeuron.setInput(neuronInput=testInput[0])
            andNeuron.run()
            self.assertEqual(int(andNeuron.output), testInput[1])

    def testNeuronINVERTGATE(self):
        """ Testing of the perceptron classe with the INVERT/NOT gate with every possible input"""
        notNeuron = Neuron(weights=[-1], bias=0)

        inputInvertNeuron = [[[1], 0],
                                 [[0], 1]]

        for testInput in inputInvertNeuron:
            notNeuron.setInput(neuronInput=testInput[0])
            notNeuron.run()
            self.assertNotEqual(notNeuron.output, testInput[1])

        """
        De notNeuron gate is alleen maar mogelijk bij het afronden van de output floats naar integers.
        """
        notNeuron = Neuron(weights=[-110], bias=100)
        for testInput in inputInvertNeuron:
            notNeuron.setInput(neuronInput=testInput[0])
            notNeuron.run()
            self.assertEqual(int(notNeuron.output), testInput[1])

    def testNeuronORGATE(self):
        """ Testing of the neuron classe with the OR gate with every possible input"""
        orNeuron = Neuron(weights=[1, 1], bias=-1)

        inputOrNeuron = [[[1, 1], 1],
                             [[1, 0], 1],
                             [[0, 1], 1],
                             [[0, 0], 0]]

        for testInput in inputOrNeuron:
            orNeuron.setInput(neuronInput=testInput[0])
            orNeuron.run()
            self.assertNotEqual(orNeuron.output, testInput[1])

        """
        De orNeuron gate is alleen maar mogelijk bij het afronden van de output floats naar integers.
        """
        orNeuron = Neuron(weights=[100, 100], bias=-50)
        for testInput in inputOrNeuron:
            orNeuron.setInput(neuronInput=testInput[0])
            orNeuron.run()
            self.assertEqual(int(orNeuron.output), testInput[1])

    """ 2.a toelichting
    Waarom werken de INVERT-, AND- en OR-poorten neurons niet met de zelfde parameters als bij de perceptron?
    Omdat er nu als output geen 0/1 meer uit komt. Maar een getal tussen de 0 en 1. Met andere parameters is het 
    mogelijk om heel dicht bij de 1 of 0 te komen. Maar zonder afronden krijg je er geen 0 of 1 uit. 
    Het is wel mogelijk om 1.0 er uit te halen, maar de 0.0 is onmogelijk.
    """

    def testNeuronNORGATE(self):
        """2.B"""
        """ Testing of the neuron classe with the NOR gate with every possible input"""
        norNeuron = Neuron(weights=[-100, -100, -100], bias=100)

        inputNorNeuron = [[[0, 0, 0], 1],
                          [[1, 0, 0], 0],
                          [[1, 1, 0], 0],
                          [[1, 1, 1], 0],
                          [[0, 1, 1], 0],
                          [[0, 0, 1], 0],
                          [[0, 1, 0], 0]]

        """
        Ook de norNeuron gate is alleen maar mogelijk bij het afronden van de output floats naar integers.
        """
        for testInput in inputNorNeuron:
            norNeuron.setInput(neuronInput=testInput[0])
            norNeuron.run()
            self.assertEqual(int(norNeuron.output), testInput[1])