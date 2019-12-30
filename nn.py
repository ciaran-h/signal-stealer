import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import cv2
import uuid
import random
import matplotlib.bezier

class FeedForwardNN2:
    '''
    A simple implementation of a feed forward neural network.
    '''

    def __init__(self, inputNodes, hiddenNodes1, hiddenNodes2, outputNodes, learningRate=1.0, seed=None):
        '''
        Create a new feed forward neural network, with the specified
        number of nodes initialized with random values.
        '''
        # Keep results consistent
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(hash(uuid.uuid4()) % (2**32 - 1))

        self._learningRate = learningRate

        self._inputNodes = inputNodes
        self._outputNodes = outputNodes

        self._numOfNodes = [inputNodes, hiddenNodes1, hiddenNodes2, outputNodes]

        # Initialize the weights with random values
        self._weights = []
        self._weights.append(np.random.rand(inputNodes, hiddenNodes1) * 2.0 - 1.0)
        self._weights.append(np.random.rand(hiddenNodes1, hiddenNodes2) * 2.0 - 1.0)
        self._weights.append(np.random.rand(hiddenNodes2, outputNodes) * 2.0 - 1.0)
        self._bias = []
        self._bias.append(0)
        self._bias.append(0)
        self._bias.append(0)

    def _atan(self, z):
        return np.arctan(z) 

    def _atanDerivative(self, z):
        return 1.0 / (z**2 + 1.0)

    def _sigmoid(self, z):
        '''
        An activation function returning values in the
        interval (0, 1)
        '''
        #return np.arctan(z)
        #return np.tanh(z)
        return 1.0 / (1.0 + np.exp(-z))
    
    def _sigmoidDerivative(self, z):
        '''
        The derivative of the sigmoid function.
        i.e. the direction the sigmoid function increases.
        '''
        #return 1 / (z**2 + 1)
        #return 1.0 - np.tanh(z)**2
        return self._sigmoid(z) * (1.0 - self._sigmoid(z))
    
    def _relu(self, x):
        '''
        The master of stupidity
        '''
        z = np.copy(x)
        return z * (z > 0)

    def _reluDerivative(self, x):
        '''
        Does something important. I dunno Google it.
        '''
        z = np.copy(x)
        return 1.0 * (z > 0)
    
    def _leakyRelu(self, x):
        z = np.copy(x)
        return np.where(z > 0, z, z * 0.01)

    def _leakyReluDerivative(self, x):
        z = np.copy(x)
        return np.where(z > 0, 1, 0.01)

    def gradientDescent(self, inputs, target, iterations, graph=True, draw=False):
        '''
        Trains the network. But you already knew that right?
        '''
        
        # Check the the inputs and target dimensions make sense.
        assert len(inputs) == len(target), 'Each input should have a target value.'
        assert len(inputs[0]) == self._inputNodes, 'The number of inputs does not match the number of input nodes.'
        assert len(target[0]) == self._outputNodes, 'The number of outputs does not match the number of output nodes.'

        lastDrawTime = 0

        if graph:
            xAxis = []
            yAxis = []
            plt.title('Training')
            plt.xlabel('Iteration')
            plt.ylabel('Error (MSE)')
            #plt.ylim(0, 1)
            #plt.xlim(0,iterations)

        print("Training", end='')
        for i in range(iterations):
            
            if i % (iterations // 10) == 0:
                print('.', end=''); sys.stdout.flush()

            # FORWARD PASS
            # Feed the values through the network and the calculate the
            # error between the result <feed2Activated> and the target
            # result <target>
            feed1 = np.dot(inputs, self._weights[0]) + self._bias[0]
            feed1Activated = self._leakyRelu(feed1)
            feed2 = np.dot(feed1Activated, self._weights[1]) + self._bias[1]
            feed2Activated = self._leakyRelu(feed2)
            feed3 = np.dot(feed2Activated, self._weights[2]) + self._bias[2]
            feed3Activated = self._atan(feed3)

            
            # BACK PROPAGATION
            error = target - feed3Activated
            # Calculate the gradient for the weights
            layers3to2Gradient = np.dot(
                feed2Activated.T,
                error * self._atanDerivative(feed3)
            )
            layers3to2Bias = error * self._atanDerivative(feed3)

            layers3to1Gradient = np.dot(
                feed1Activated.T,
                np.dot(
                    error * self._atanDerivative(feed3),
                    self._weights[2].T
                ) * self._leakyReluDerivative(feed2)
            )
            layers3to1Bias = np.dot(
                    error * self._atanDerivative(feed3),
                    self._weights[2].T
                ) * self._leakyReluDerivative(feed2)

            layers3to0Gradient = np.dot(
                inputs.T,
                np.dot(
                    np.dot(
                        error * self._atanDerivative(feed3),
                        self._weights[2].T
                    ) * self._leakyReluDerivative(feed2),
                    self._weights[1].T
                ) * self._leakyReluDerivative(feed1)
            )
            layers3to0Bias = np.dot(
                    np.dot(
                        error * self._atanDerivative(feed3),
                        self._weights[2].T
                    ) * self._leakyReluDerivative(feed2),
                    self._weights[1].T
                ) * self._leakyReluDerivative(feed1)

            # Update the weights
            self._weights[2] += layers3to2Gradient * self._learningRate
            self._weights[1] += layers3to1Gradient * self._learningRate
            self._weights[0] += layers3to0Gradient * self._learningRate
            self._bias[2] += layers3to2Bias * self._learningRate
            self._bias[1] += layers3to1Bias * self._learningRate
            self._bias[0] += layers3to0Bias * self._learningRate

            # Plot the graph every 1/30th of a second
            if time.time() - lastDrawTime > (1.0 / 30.0):
                if graph:
                    xAxis.append(i)
                    yAxis.append(self.meanSquaredError(error))
                    plt.plot(xAxis, yAxis, color='red')
                    plt.pause(0.0001)
                if draw:
                    self._draw()
                lastDrawTime = time.time()
        
        if graph:
            plt.plot(xAxis, yAxis, color='black')
            plt.show()
        
        print(' Done!')

    def prediction(self, inputs):
        '''
        Feeds the input through the neural network and returns
        what is the predicted result.
        i.e. Forward propagation
        '''

        x1 = self._sigmoid(np.dot(inputs, self._weights[0]))
        x2 = self._sigmoid(np.dot(x1, self._weights[1]))
        x3 = self._sigmoid(np.dot(x2, self._weights[2]))

        return x3
    
    def meanSquaredError(self, error):
        return np.dot(error[0], error[0].T) / len(error[0])

    def _draw(self):

        width, height = 1024, 1024

        img = np.full((width, height, 3), 255, np.uint8)

        buffer = 3
        minLineWidth = 1

        maxNumOfNodes = max(self._numOfNodes)
        nodeRadius = height // (buffer * 2 * maxNumOfNodes)

        maxLineWidth = nodeRadius // 2

        verticalSpacing = buffer * nodeRadius * 2
        horizontalSpacing = width // len(self._numOfNodes)

        # Draw the weights
        maxWeight = max(np.amax(w) for w in self._weights)
        minWeight = min(np.amin(w) for w in self._weights)
        
        for layer in range(len(self._weights)):
            w = self._weights[layer]
            leftX = horizontalSpacing * layer + horizontalSpacing // 2
            rightX = horizontalSpacing * (layer + 1) + horizontalSpacing // 2
            for leftNode in range(len(w)):
                for rightNode in range(len(w[leftNode])):
                    leftY = verticalSpacing * leftNode + verticalSpacing + nodeRadius * 2
                    rightY = verticalSpacing * rightNode + verticalSpacing + nodeRadius * 2
                    strength = (w[leftNode][rightNode] - minWeight) / (maxWeight - minWeight)
                    width = int((maxLineWidth - minLineWidth) * strength + minLineWidth)

                    cv2.line(img, (leftX, leftY), (rightX, rightY), (255,255 * (1-strength),255 * (1-strength)), width, cv2.LINE_AA)

        # Draw the nodes
        for layer in range(len(self._numOfNodes)):
            for node in range(self._numOfNodes[layer]):
                x = horizontalSpacing * layer + horizontalSpacing // 2
                y = verticalSpacing * node + verticalSpacing + nodeRadius * 2
                cv2.circle(img, (x, y), nodeRadius, (15,15,15), cv2.FILLED, cv2.LINE_AA)

        # img = cv2.line(img, (0,0), (random.randint(0, 511) , random.randint(0, 511)), (255,0,0), 5)
        cv2.imshow('Training Visualization', img)
        cv2.waitKey(20)











# Based on: https://towardsdatascience.com/a-step-by-step-implementation-of-gradient-descent-and-backpropagation-d58bda486110
class FeedForwardNN:
    '''
    A simple implementation of a feed forward neural network.
    '''

    def __init__(self, inputNodes, hiddenNodes, outputNodes, learningRate=1.0, seed=314159):
        '''
        Create a new feed forward neural network, with the specified
        number of nodes initialized with random values.
        '''
        # Keep results consistent
        np.random.seed(seed)

        self._learningRate = learningRate

        self._inputNodes = inputNodes
        self._hiddenNodes = hiddenNodes
        self._outputNodes = outputNodes

        # Initialize the weights with random values
        self.inputToHiddenWeights   = np.random.rand(inputNodes, hiddenNodes)
        self.hiddenToOutputWeights  = np.random.rand(hiddenNodes, outputNodes)

    def _sigmoid(self, z):
        '''
        An activation function returning values in the
        interval (0, 1)
        '''
        #return np.tanh(z)
        return 1.0 / (1.0 + np.exp(-z))
    
    def _sigmoidDerivative(self, z):
        '''
        The derivative of the sigmoid function.
        i.e. the direction the sigmoid function increases.
        '''
        #return 1.0 - np.tanh(z)**2
        return self._sigmoid(z) * (1.0 - self._sigmoid(z))
    
    def gradientDescent(self, inputs, target, iterations):
        '''
        Trains the network.
        '''

        # Check the the inputs and target dimensions make sense.
        assert len(inputs) == len(target), 'Each input should have a target value.'
        assert len(inputs[0]) == self._inputNodes, 'The number of inputs does not match the number of input nodes.'
        assert len(target[0]) == self._outputNodes, 'The number of outputs does not match the number of output nodes.'

        xAxis = []
        yAxis = []

        print("Training", end='')
        for i in range(iterations):

            if i % (iterations // 10) == 0:
                print('.', end='')

            # Feed the values through the network and the calculate the
            # error between the result <feed2Activated> and the target
            # result <target>
            feed1 = np.dot(inputs, self.inputToHiddenWeights)
            feed1Activated = self._sigmoid(feed1)
            feed2 = np.dot(feed1Activated, self.hiddenToOutputWeights)
            feed2Activated = self._sigmoid(feed2)
            error = target - feed2Activated

            xAxis.append(i)
            yAxis.append(np.dot(error[0], error[0].T))

            # Calculate the gradient for the weights
            hiddenToOutputGradient = np.dot(
                feed1Activated.T,
                error * self._sigmoidDerivative(feed2)
            )

            inputToHiddenGradient = np.dot(
                inputs.T,
                np.dot(
                    error * self._sigmoidDerivative(feed2),
                    self.hiddenToOutputWeights.T
                ) * self._sigmoidDerivative(feed1)
            )

            # Update the weights
            self.inputToHiddenWeights   += inputToHiddenGradient * self._learningRate
            self.hiddenToOutputWeights  += hiddenToOutputGradient * self._learningRate
        
        print(' Done!')
        plt.plot(xAxis, yAxis)
        plt.show()

    def prediction(self, inputs):
        '''
        Feeds the input through the neural network and returns
        what is the predicted result.
        i.e. Forward propagation
        '''

        x1 = self._sigmoid(np.dot(inputs, self.inputToHiddenWeights))
        x2 = self._sigmoid(np.dot(x1, self.hiddenToOutputWeights))

        return x2