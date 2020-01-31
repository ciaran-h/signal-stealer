import numpy as np
import matplotlib.pyplot as plt
import time
import sys
import cv2
import uuid
import random
import math
import act
from abc import ABC, abstractmethod

def _randomWeights(rows, cols):
    return np.random.rand(rows, cols) * 2.0 - 1.0

def _meanSquared(vector):
    return np.dot(vector[0], vector[0].T) / len(vector[0])

class SimpleFFNNBuilder():

    _nodesPerLayer = []
    _activationFunctions = []

    def __init__(self, learningRate=1, seed=None):
        self._learningRate = learningRate
        self._seed = seed

    def addLayers(self, *nodesPerLayer, actFun=act.ReluAF):
        for i in range(len(nodesPerLayer)):
            self._nodesPerLayer.append(nodesPerLayer[i])
            self._activationFunctions.append(actFun)
        return self
    
    def build(self):

        assert len(self._nodesPerLayer) >= 2, \
            "Need at least two layers (i.e. an input and output layer)."

        result = SimpleFFNN(
            self._nodesPerLayer,
            self._activationFunctions,
            learningRate=self._learningRate,
            seed=self._seed
        )

        return result


class SimpleFFNN:
    
    # Visualization
    _fps = 2
    _frame = 0
    _inputRange, _outputRange = (-1,1), (-1,1)

    # Neural Network
    _weights = []
    _act = []
    _bias = []
    _inputs = _target = _feed = _feedAct = None

    # Graphing
    _xAxisOverview, _yAxisOverview = [], []
    _xAxisLive, _yAxisLive = [], []
    _targetXValues = 50
    _expStep, _step = 1, 100
    _liveValues = 100

    def __init__(self, nodesPerLayer, activationFunctions, learningRate=1, seed=None):
        
        # Keep results consistent if wanted
        if seed is not None:
            np.random.seed(seed)
        else:
            np.random.seed(hash(uuid.uuid4()) % (2**32 - 1))

        self._learningRate = learningRate
        self._numOfLayers = len(nodesPerLayer)
        self._numOfWeights = self._numOfLayers-1
        self._nodesPerLayer = nodesPerLayer
    
        for i in range(self._numOfWeights):
            nodesInCurLayer = nodesPerLayer[i]
            nodesInNextLayer = nodesPerLayer[i+1]
            self._weights.append(_randomWeights(nodesInCurLayer, nodesInNextLayer))
            self._bias.append(1)
        
        self._act = activationFunctions

    def _gradientDescentHelper(self, layer, loss):

        # BASE CASE: output layer
        if layer == self._numOfLayers-1:
            return loss * self._act[layer-1].computeDer(self._feed[layer])
        
        # RECURSION
        result = np.dot(
            self._gradientDescentHelper(layer+1, loss),
            self._weights[layer].T
        ) * self._act[layer-1].computeDer(self._feed[layer])
        
        return result
    
    def _gradientDescent(self, layer, loss):

        gradient = self._gradientDescentHelper(layer+1, loss)
        weightDelta = np.dot(
            self._feedAct[layer].T,
            gradient
        )
        # TODO: not sure if this is a correct implementation
        biasDelta = sum(sum(gradient))

        return weightDelta, biasDelta

    def backPropagation(self, loss):

        for k in range(self._numOfWeights):
            weightDelta, biasDelta = self._gradientDescent(k, loss)
            self._weights[k] += weightDelta * self._learningRate
            self._bias[k] += biasDelta * self._learningRate

    def forwardPropagation(self, inputs):

        self._feed[0] = self._feedAct[0] = inputs
        for i in range(1, self._numOfLayers):
            self._feed[i] = np.dot(self._feedAct[i-1], self._weights[i-1]) + self._bias[i-1]
            self._feedAct[i] = self._act[i-1].compute(self._feed[i])

        return self._feedAct[-1]

    def setTrainingData(self, inputs, target, inputRange=None, outputRange=None):

        if inputRange is not None: self._inputRange = inputRange
        if outputRange is not None: self._outputRange = outputRange

        self._numOfSamples = len(inputs)

        self._inputs = inputs
        self._target = target

        self._feed = []
        self._feedAct = []

        # Initialize the feeds
        self._feed.append(np.zeros((len(inputs), len(inputs[0]))))
        self._feedAct.append(np.zeros((len(inputs), len(inputs[0]))))
        for i in range(1, self._numOfLayers):
            shape = (self._nodesPerLayer[i] * self._numOfSamples, len(self._weights[i-1]))
            self._feed.append(np.zeros(shape))
            self._feedAct.append(np.zeros(shape))
        

    def train(self, epoch, graph=True, showOutput=False, showWeights=False):

        # Ensure training data has been supplied and makes sense
        assert self._inputs is not None and self._target is not None, \
            "No training data. Make sure you call 'setTrainingData' before training."
        assert len(self._inputs) == len(self._target), \
            'Each input should have a target value.'
        
        # Check that the network dimensions make sense
        assert len(self._inputs[0]) == self._nodesPerLayer[0], \
            'The number of input values does not match the number of input nodes.'
        assert len(self._target[0]) == self._nodesPerLayer[-1], \
            'The number of output values does not match the number of output nodes.'

        lastDrawTime = 0
        if graph: self._initGraph()

        for i in range(epoch):
            # Update the learning rate
            # TODO: remake this -- give user more control
            if (i + 1) % (epoch // 4) == 0:
                print('Learning rate reduced from ', self._learningRate, end='')
                self._learningRate *= 0.5
                print(' to ', self._learningRate)

            # Propagate the input through the network and get
            # the result
            result = self.forwardPropagation(self._inputs)
            # Determine the loss between the actual result and the 
            # target result
            loss = self._target - result
            # Update the weights to reduce this loss
            self.backPropagation(loss)

            # Draw any visualizations
            if graph: self._graphPoint(i, _meanSquared(loss))
            if time.time() - lastDrawTime > (1.0 / self._fps):
                if showOutput: self._visualizeOutput()
                if showWeights: self._visualizeWeights()
                if graph: self._drawGraph()
                lastDrawTime = time.time()
    
    def _initGraph(self):

        self._xAxisOverview.clear()
        self._yAxisOverview.clear()
        self._xAxisLive.clear()
        self._yAxisLive.clear()
        self._expStep = 1

    def _graphPoint(self, x, y):

        if x % self._step == 0:
            self._xAxisLive.append(x)
            self._yAxisLive.append(y)
            if len(self._xAxisLive) > self._liveValues:
                self._xAxisLive.pop(0)
                self._yAxisLive.pop(0)
        if x % self._expStep == 0:
            # Add the point to the graph
            self._xAxisOverview.append(x)
            self._yAxisOverview.append(y)
            # If the list has grown to two times the
            # target size, halve it
            if len(self._xAxisOverview) % (self._targetXValues * 2) == 0:
                #print('Halving array...')
                # Remove every other element
                self._xAxisOverview = self._xAxisOverview[::2]
                self._yAxisOverview = self._yAxisOverview[::2]
                self._expStep *= 2
    
    def _drawGraph(self):

        plt.clf()

        plt.subplot(3, 1, 1)
        plt.title('Training Overview')
        plt.xlabel('Iteration')
        plt.ylabel('Error (MSE)')
        plt.plot(self._xAxisOverview, self._yAxisOverview, color='red')

        plt.subplot(3, 1, 3)
        plt.title('Real-time Training')
        plt.xlabel('Iteration')
        plt.ylabel('Error (MSE)')
        plt.plot(self._xAxisLive, self._yAxisLive, color='red')

        plt.pause(0.0001)

    def _visualizeOutput(self):

        assert self._nodesPerLayer[0] == 2, \
            'There must be exactly 2 nodes in the input layer ' \
            'for it to be visualized.'
        assert self._nodesPerLayer[-1] <= 3, \
            'There must be 3 or less nodes in the output layer ' \
            'for it to be visualized.'

        size = 25
        resizedSize = 512
        width = size * 2 + 1
        img = np.full((width, width, 3), 255, np.uint8)
        
        def _normalize(v, r):
            v = max(min(v, r[1]), r[0])
            return (v - r[0]) / (r[1]-r[0])

        def _expand(v, r):
            v = max(min(v, 1), 0)
            return v * (r[1]-r[0]) + r[0]

        # Feeds values in [-1,1] to the neural network and gets their values
        inputs = np.array([[
                    _expand(_normalize(x, (0,width-1)), self._inputRange),
                    _expand(_normalize(y, (0,width-1)), self._inputRange)]
                    for y in range(width) for x in range(width)])
        prevInputs, prevTarget = self._inputs, self._target
        self.setTrainingData(inputs, None)
        values = self.forwardPropagation(inputs)
        self.setTrainingData(prevInputs, prevTarget)
        # TODO: inlude max and min for sample targets
        #largest = max(max(value) for value in values)
        #smallest = min(min(value) for value in values)

        def _getPixelColor(v):
            c = [255, 255, 255]
            for i in range(len(v)):
                c[i] = round(_normalize(v[i], self._outputRange) * 255)
            return tuple(c)

        # Set the pixel colors
        for x in range(width):
            for y in range(width):
                i = x * width + y
                img[x][y] = _getPixelColor(values[i])
        
        #cv2.imwrite("C:\\Users\\Ciaran Hogan\\Desktop\\nn\\frame_"+str(self._frame)+".png", img)
        #self._frame = self._frame + 1

        # Ensure we return the same subset everytime
        # TODO: only compute this once
        rng = random.Random(0)
        samples = rng.choices([i for i in range(len(self._inputs))], k=(size*size)//10)
        
        # Resize the image
        resized = cv2.resize(img, (resizedSize, resizedSize))
        
        # Draw the target samples on the resized image
        for i in samples:
            
            # The assert allows us to do this
            # with reasonable assurance that it will
            # work correctly
            x, y = self._inputs[i]
            nx, ny = (x-self._inputRange[0])/(self._inputRange[1]-self._inputRange[0]), (y-self._inputRange[0])/(self._inputRange[1]-self._inputRange[0])
            px, py = int(round(nx*(resizedSize-1))), int(round(ny*(resizedSize-1)))
            # Determine the color of the pixel
            rectSize = 4
            bgRectStart, bgRectEnd = (px-rectSize, py-rectSize), (px+rectSize, py+rectSize)
            cRectStart, cRectEnd = (px-rectSize+1, py-rectSize+1), (px+rectSize-1, py+rectSize-1)
            cv2.rectangle(resized, bgRectStart, bgRectEnd, (255,255,255), thickness=cv2.FILLED)
            cv2.rectangle(resized, cRectStart, cRectEnd, _getPixelColor(self._target[i]), thickness=cv2.FILLED, lineType=cv2.LINE_AA)
            #resized[py][px] = _getPixelColor(self._target[i])
        
        # Show the image
        cv2.imshow('Output Visualization', resized)
        cv2.waitKey(20)
    
    def _visualizeWeights(self):

        windowSize = 512
        maxWindowSize = windowSize * 2

        maxNumOfNodes = max(self._nodesPerLayer)

        ratio = self._numOfLayers / maxNumOfNodes
        if ratio >= 0:
            width = windowSize
            height = min(maxWindowSize, round(width * 1 / ratio))
        else:
            height = windowSize
            width = min(maxWindowSize, round(height * ratio))
        img = np.full((height, width, 3), 255, np.uint8)

        buffer = 4
        minLineWidth = 1
        orange = (233, 121, 67)
        blue = (67, 125, 233)
        grey = (34, 40, 49)
        nodeRadius = height // (buffer * 2 * maxNumOfNodes)
        maxLineWidth = max(minLineWidth, nodeRadius // 4)
        verticalSpacing = buffer * nodeRadius * 2
        horizontalSpacing = width // len(self._nodesPerLayer)

        # Draw the weights
        maxWeight = max(np.amax(w) for w in self._weights)
        minWeight = min(np.amin(w) for w in self._weights)

        for layer in range(len(self._weights)):
            w = self._weights[layer]
            leftX = horizontalSpacing * layer + horizontalSpacing // 2
            rightX = horizontalSpacing * (layer + 1) + horizontalSpacing // 2
            for leftNode in range(len(w)):
                for rightNode in range(len(w[leftNode])):
                    leftY = verticalSpacing * leftNode + verticalSpacing // 2 - nodeRadius // 2
                    rightY = verticalSpacing * rightNode + verticalSpacing // 2 - nodeRadius // 2
                    strength = (w[leftNode][rightNode] -
                                minWeight) / (maxWeight - minWeight)
                    lineWidth = int((maxLineWidth - minLineWidth)
                                * strength + minLineWidth)
                    if w[leftNode][rightNode] >= 0:
                        color = tuple([int((1 - strength) * (255 - x) + x) for x in orange])
                    else:
                        color = tuple([int((1 - strength) * (255 - x) + x) for x in blue])
                    cv2.line(img, (leftX, leftY), (rightX, rightY), color, lineWidth, cv2.LINE_AA)

        # Draw the nodes
        for layer in range(len(self._nodesPerLayer)):
            for node in range(self._nodesPerLayer[layer]):
                x = horizontalSpacing * layer + horizontalSpacing // 2
                y = verticalSpacing * node + verticalSpacing // 2 - nodeRadius // 2
                cv2.circle(img, (x, y), nodeRadius, grey, cv2.FILLED, cv2.LINE_AA)
        
        cv2.imshow('Weight Visualization', img)
        cv2.waitKey(20)