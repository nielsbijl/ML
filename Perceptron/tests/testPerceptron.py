import unittest
from Perceptron.Perceptron import *


class TestPerceptron(unittest.TestCase):

    def testPerceptronANDGATE(self):
        """ Het testen van de perceptron classe met de AND gate met alle verschillende inputs"""
        andPerceptron = Perceptron(weights=[1, 1], bias=2)

        andPerceptron.setInput(input=[1, 1])
        andPerceptron.run()
        self.assertEqual(andPerceptron.ouput, 1)

        andPerceptron.setInput(input=[0, 1])
        andPerceptron.run()
        self.assertEqual(andPerceptron.ouput, 0)

        andPerceptron.setInput(input=[1, 0])
        andPerceptron.run()
        self.assertEqual(andPerceptron.ouput, 0)

        andPerceptron.setInput(input=[0, 0])
        andPerceptron.run()
        self.assertEqual(andPerceptron.ouput, 0)

    def testPerceptronINVERTGATE(self):
        """ Het testen van de perceptron classe met de INVERT gate met alle verschillende inputs"""
        notPerceptron = Perceptron(input=[1], weights=[-1], bias=0)

        notPerceptron.setInput(input=[1])
        notPerceptron.run()
        self.assertEqual(notPerceptron.ouput, 0)

        notPerceptron.setInput(input=[0])
        notPerceptron.run()
        self.assertEqual(notPerceptron.ouput, 1)

    def testPerceptronORGATE(self):
        """ Het testen van de perceptron classe met de OR gate met alle verschillende inputs """
        orPerceptron = Perceptron(weights=[1, 1], bias=1)

        orPerceptron.setInput(input=[1, 1])
        orPerceptron.run()
        self.assertEqual(orPerceptron.ouput, 1)

        orPerceptron.setInput(input=[1, 0])
        orPerceptron.run()
        self.assertEqual(orPerceptron.ouput, 1)

        orPerceptron.setInput(input=[0, 1])
        orPerceptron.run()
        self.assertEqual(orPerceptron.ouput, 1)

        orPerceptron.setInput(input=[0, 0])
        orPerceptron.run()
        self.assertEqual(orPerceptron.ouput, 0)

    def testPerceptronNORGATE(self):
        """ Het testen van de perceptron classe met de NOR gate met alle verschillende inputs """
        norPerceptron = Perceptron(weights=[-1, -1, -1], bias=0)

        norPerceptron.setInput(input=[0, 0, 0])
        norPerceptron.run()
        self.assertEqual(norPerceptron.ouput, 1)

        norPerceptron.setInput(input=[1, 0, 0])
        norPerceptron.run()
        self.assertEqual(norPerceptron.ouput, 0)

        norPerceptron.setInput(input=[1, 1, 0])
        norPerceptron.run()
        self.assertEqual(norPerceptron.ouput, 0)

        norPerceptron.setInput(input=[1, 1, 1])
        norPerceptron.run()
        self.assertEqual(norPerceptron.ouput, 0)

        norPerceptron.setInput(input=[0, 1, 1])
        norPerceptron.run()
        self.assertEqual(norPerceptron.ouput, 0)

        norPerceptron.setInput(input=[0, 0, 1])
        norPerceptron.run()
        self.assertEqual(norPerceptron.ouput, 0)

    def testPerceptronPARTY(self):
        """ Het testen van de perceptron classe met de NOR gate met alle verschillende inputs """
        partyPerceptron = Perceptron(weights=[0.6, 0.3, 0.2], bias=0.4)

        partyPerceptron.setInput(input=[1, 1, 0])
        partyPerceptron.run()
        self.assertEqual(partyPerceptron.ouput, 1)

        partyPerceptron.setInput(input=[1, 0, 1])
        partyPerceptron.run()
        self.assertEqual(partyPerceptron.ouput, 1)

        partyPerceptron.setInput(input=[1, 0, 0])
        partyPerceptron.run()
        self.assertEqual(partyPerceptron.ouput, 1)

        partyPerceptron.setInput(input=[0, 1, 1])
        partyPerceptron.run()
        self.assertEqual(partyPerceptron.ouput, 1)

        partyPerceptron.setInput(input=[0, 0, 1])
        partyPerceptron.run()
        self.assertEqual(partyPerceptron.ouput, 0)

        partyPerceptron.setInput(input=[0, 1, 0])
        partyPerceptron.run()
        self.assertEqual(partyPerceptron.ouput, 0)


if __name__ == '__main__':
    unittest.main()
