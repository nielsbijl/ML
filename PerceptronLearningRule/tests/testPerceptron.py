import unittest
from PerceptronLearningRule.Perceptron import *


class TestPerceptron(unittest.TestCase):

    def testPerceptronANDGATE(self):
        """
            ########## 3.A ##########
        """

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
        epochs = 100
        while epochs > 0:
            for i in range(len(truthTable)):
                andPerceptron.setInput(truthTable[i][0])
                andPerceptron.update(truthTable[i][1], learningRate=0.8)
                print(epochs)
            epochs -= 1

        """ Proving the andPerceptron is now valid """
        predOutp = []
        for i in range(len(truthTable)):
            andPerceptron.setInput(truthTable[i][0])
            andPerceptron.run()
            predOutp.append(andPerceptron.output)
        self.assertEqual(testOutp, predOutp)

        """Wat zijn de uiteindelijke parameters van de perceptron?"""
        print("AND final parameters:"+ andPerceptron.__str__())
        print("AND MSE:", andPerceptron.error())

    def testPerceptronXORGATE(self):
        """
            ########## 3.B ##########
        """

        """ Oefening 4.3 - Perceptron learning rule: XOR"""
        truthTable = [
            [[1, 1], 0],
            [[1, 0], 1],
            [[0, 1], 1],
            [[0, 0], 0],
            [[1, 1], 0],
            [[1, 0], 1]
        ]
        testOutp = [row[1] for row in truthTable]

        xorPerceptron = Perceptron(weights=[1, 1], bias=-1)

        """ Proving the xorPerceptron is invalid """
        predOutp = []
        for i in range(len(truthTable)):
            xorPerceptron.setInput(truthTable[i][0])
            xorPerceptron.run()
            predOutp.append(xorPerceptron.output)
        self.assertNotEqual(testOutp, predOutp)

        """ Learning the xorPerceptron """
        for x in range(10):
            for i in range(len(truthTable)):
                xorPerceptron.setInput(truthTable[i][0])
                xorPerceptron.update(truthTable[i][1], learningRate=0.1)

        """ Proving the andPerceptron is now still invalid """
        predOutp = []
        for i in range(len(truthTable)):
            xorPerceptron.setInput(truthTable[i][0])
            xorPerceptron.run()
            predOutp.append(xorPerceptron.output)
        self.assertNotEqual(testOutp, predOutp)

        """Wat zijn de uiteindelijke parameters van de perceptron?"""
        print("XOR final parameters:" + xorPerceptron.__str__())
        print("XOR MSE:", xorPerceptron.error())

    """
    UITLEG: Waarom werkt het niet? 
    
    Een enkele perceptron heeft geen mogelijkheid om de XOR functionaliteit te 
    krijgen, omdat dit geen linear scheidbaar probleem is. Hier is een netwerk voor nodig. Het is jammer genoeg met de step 
    functie niet mogelijk om een netwerk te trainen. Waardoor het niet mogelijk is om backpropagation te gebruiken 
    bij een netwerk van perceptrons. Dit kan gelukkig wel bij sigmoid neuron omdat de sigmoid functie dit wel toelaat.
    
    """


if __name__ == '__main__':
    unittest.main()
