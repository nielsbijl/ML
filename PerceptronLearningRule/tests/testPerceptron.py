import unittest
from PerceptronLearningRule.Perceptron import *


class TestPerceptron(unittest.TestCase):

    def testPerceptronANDGATE(self):
        """ Oefening 4.1 - PerceptronLearningRule: AND"""
        truthTable = [
            [[1, 1], 1],
            [[1, 0], 0],
            [[0, 1], 0],
            [[0, 0], 0],
            [[1, 1], 1],
            [[1, 0], 0],
            [[0, 1], 0],
            [[0, 0], 0],
            [[1, 1], 1],
            [[1, 0], 0],
        ]
        testOutp = [row[1] for row in truthTable]

        """ Creating the invalid andPerceptron """
        andPerceptron = Perceptron([-0.5, 0.5], -1.5)

        """ Proving the andPerceptron is invalid """
        predOutp = []
        for i in range(len(truthTable)):
            andPerceptron.setInput(truthTable[i][0])
            andPerceptron.run()
            predOutp.append(andPerceptron.output)
        self.assertNotEqual(testOutp, predOutp)

        """ Learning the andPerceptron """
        for i in range(len(truthTable)):
            andPerceptron.setInput(truthTable[i][0])
            andPerceptron.update(truthTable[i][1], learningRate=0.8)

        """ Proving the andPerceptron is now valid """
        predOutp = []
        for i in range(len(truthTable)):
            andPerceptron.setInput(truthTable[i][0])
            andPerceptron.run()
            predOutp.append(andPerceptron.output)
        self.assertEqual(testOutp, predOutp)


if __name__ == '__main__':
    unittest.main()
