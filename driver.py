import numpy as np
import string
import random
import const
import bodyCodes
import nn

'''
NETWORK STRUCTURE

    Input Layer
        Nodes: [1,maxLength * len(const.bodyCodes)]

    Hidden layer (Layer 1)
        Nodes: maxLength * len(const.bodyCodes)
        Dimensions of weighting factor matrix: [maxLength * len(const.bodyCodes),maxLength]
        Output dimension: [1,maxLength]

    Output Layer (Layer 2)
        Nodes: 1
        Dimensions of weighting factor matrix: [maxLength,1]
        Output dimension: [1,1]
'''


# Generate the testing data
testSamples = 10**3
print('Generating {} test samples...'.format(testSamples))
#testInputs, testOutputs = bodyCodes.generateSampleData(testSamples, 0.33, 0.33)
testInputs, testOutputs = bodyCodes.generateSampleData(testSamples, 0.00, 0.50)

# Create the neural network
inputNodes = len(const.bodyCodes) * const.maxNumOfActions
print('Creating a feed forward neural network with {} input node(s), '\
    '{} hidden node(s), and {} output node(s).'.format(inputNodes, inputNodes, 1))
ffnn = nn.FeedForwardNN2(inputNodes, inputNodes, 24, 1, learningRate=0.0001)

#print(ffnn.inputToHiddenWeights)
#print(ffnn.hiddenToOutputWeights)

# Train the neural network
iterations = 10**4
print('Training the network for {} iterations...'.format(iterations))
ffnn.gradientDescent(testInputs, testOutputs, iterations, graph=True, draw=True)

# Validate
validationSamples = 10**3

correctStealNoFakeOut = 0
for i in range(validationSamples):
    seq = bodyCodes.bodyCodeSequenceToIntCode(bodyCodes.randomStealWithoutFakeOut())
    isSteal = True if ffnn.prediction(seq) > 0.5 else False
    if isSteal:
        correctStealNoFakeOut += 1
stealNoFakeOutPercent = correctStealNoFakeOut / validationSamples

correctStealFakeOut = 0
for i in range(validationSamples):
    seq = bodyCodes.bodyCodeSequenceToIntCode(bodyCodes.randomStealWithFakeOut())
    isSteal = True if ffnn.prediction(seq) > 0.5 else False
    if not isSteal:
        correctStealFakeOut += 1
stealFakeOutPercent = correctStealFakeOut / validationSamples

correctNotSteal = 0
for i in range(validationSamples):
    seq = bodyCodes.bodyCodeSequenceToIntCode(bodyCodes.randomNoSteal())
    isSteal = True if ffnn.prediction(seq) > 0.5 else False
    if not isSteal:
        correctNotSteal += 1
noStealPercent = correctNotSteal / validationSamples

print('The model correctly predicted {} percent of non-steals, '\
    '{} percent of steals with a fake-out, and '\
    '{} percent of steals without a fake-out.'.format(
        round(noStealPercent * 100, 5), 
        round(stealFakeOutPercent * 100, 5),
        round(stealNoFakeOutPercent * 100, 5)
        )
)

'''
#X = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
#y = np.array([[0, 1, 1, 0]]).T

X = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [1, -1], [1, 0], [1, 1]])
y = np.array([[-1, 0, 1 , 0, 0, 1, 1, 1]]).T
ffnn = nn.FeedForwardNN2(2, 8, 6, 1, learningRate=0.000001)
ffnn.gradientDescent(X, y, 10**5, graph=False, draw=True)
for pair in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]:
    print('max(', pair, ') = ', ffnn.prediction(pair))

#print(ffnn._bias)

#print('0 xor 0 =', ffnn.prediction([0, 0]))
#print('1 xor 0 =', ffnn.prediction([1, 0]))
#print('0 xor 1 =', ffnn.prediction([0, 1]))
#print('1 xor 1 =', ffnn.prediction([1, 1]))

#sequence = bodyCodes.bodyCodeSequenceToIntCode(['RS','LS','LE','RE','RA'])
#print('Prediction is: ' + str(ffnn.prediction(sequence)))
'''