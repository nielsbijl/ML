import unittest
from Backpropagation.Neuron import *


class testNeuron(unittest.TestCase):
    def testANDgate(self):
        truthTable = [
            [[0, 0], 0],
            [[1, 0], 0],
            [[0, 1], 0],
            [[1, 1], 1],
        ]
        andNeuron = Neuron(weights=[-0.5, 0.5], bias=1.5)

        # Laten zien dat de andNeuron nu nog niet werkt
        expectedOutput = [row[1] for row in truthTable]
        output = []
        for row in truthTable:
            andNeuron.setInput(row[0])
            output.append(andNeuron.run())
        for i in range(len(expectedOutput)):
            self.assertNotAlmostEqual(expectedOutput[i], output[i], delta=0.1)

        # De and neuron trainen
        for epoch in range(10000):
            for row in truthTable:
                andNeuron.setInput(row[0])
                andNeuron.run()
                andNeuron.setError([row[1]])
                andNeuron.backPropagation(1)
                andNeuron.update()

        # Laten zien dat de end neuron nu wel werkt
        output = []
        for row in truthTable:
            andNeuron.setInput(row[0])
            output.append(andNeuron.run())
        for i in range(len(expectedOutput)):
            self.assertAlmostEqual(expectedOutput[i], output[i], delta=0.1)


if __name__ == '__main__':
    unittest.main()
