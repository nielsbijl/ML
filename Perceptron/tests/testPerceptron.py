import unittest
from Perceptron.Perceptron import *


class TestPerceptron(unittest.TestCase):

    def testPerceptronANDGATE(self):
        """ Testing of the perceptron classe with the AND gate with every possible input"""
        andPerceptron = Perceptron(weights=[1, 1], bias=2)

        inputAndPerceptron = [[[1, 1], 1],
                            [[1, 0], 0],
                            [[0, 1], 0],
                            [[0, 0], 0]]

        for testInput in inputAndPerceptron:
            andPerceptron.setInput(perceptronInput=testInput[0])
            andPerceptron.run()
            self.assertEqual(andPerceptron.output, testInput[1])

    def testPerceptronINVERTGATE(self):
        """ Testing of the perceptron classe with the INVERT/NOT gate with every possible input"""
        notPerceptron = Perceptron(weights=[-1], bias=0)

        inputInvertPerceptron = [[[1], 0],
                                 [[0], 1]]

        for testInput in inputInvertPerceptron:
            notPerceptron.setInput(perceptronInput=testInput[0])
            notPerceptron.run()
            self.assertEqual(notPerceptron.output, testInput[1])

    def testPerceptronORGATE(self):
        """ Testing of the perceptron classe with the OR gate with every possible input"""
        orPerceptron = Perceptron(weights=[1, 1], bias=1)

        inputOrPerceptron = [[[1, 1], 1],
                             [[1, 0], 1],
                             [[0, 1], 1],
                             [[0, 0], 0]]

        for testInput in inputOrPerceptron:
            orPerceptron.setInput(perceptronInput=testInput[0])
            orPerceptron.run()
            self.assertEqual(orPerceptron.output, testInput[1])

    def testPerceptronNORGATE(self):
        """ Testing of the perceptron classe with the NOR gate with every possible input"""
        norPerceptron = Perceptron(weights=[-1, -1, -1], bias=0)

        inputNorPerceptron = [[[0, 0, 0], 1],
                              [[1, 0, 0], 0],
                              [[1, 1, 0], 0],
                              [[1, 1, 1], 0],
                              [[0, 1, 1], 0],
                              [[0, 0, 1], 0],
                              [[0, 1, 0], 0]]

        for testInput in inputNorPerceptron:
            norPerceptron.setInput(perceptronInput=testInput[0])
            norPerceptron.run()
            self.assertEqual(norPerceptron.output, testInput[1])


    def testPerceptronPARTY(self):
        """ Testing of the perceptron classe with the PARTY gate with every possible input"""
        partyPerceptron = Perceptron(weights=[0.6, 0.3, 0.2], bias=0.4)

        inputPartyPerceptron = [[[0, 0, 0], 0],
                                [[1, 0, 0], 1],
                                [[1, 1, 0], 1],
                                [[1, 1, 1], 1],
                                [[0, 1, 1], 1],
                                [[0, 0, 1], 0],
                                [[0, 1, 0], 0]]

        for testInput in inputPartyPerceptron:
            partyPerceptron.setInput(perceptronInput=testInput[0])
            partyPerceptron.run()
            self.assertEqual(partyPerceptron.output, testInput[1])


if __name__ == '__main__':
    unittest.main()
