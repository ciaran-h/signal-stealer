import numpy as np
import string
import random
import const
import bodyCodes
import nn
import os
import math

os.system("cls")

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

'''
# Generate the testing data
testSamples = 2500
print('Generating {} test samples...'.format(testSamples))
#testInputs, testOutputs = bodyCodes.generateSampleData(testSamples, 0.33, 0.33)
testInputs, testOutputs = bodyCodes.generateSampleData(testSamples, 0.33, 0.33)

# Create the neural network
inputNodes = len(const.bodyCodes) * const.maxNumOfActions
print('Creating a feed forward neural network...')
ffnn = nn.SimpleFFNN(inputNodes, 32, 16, 16, 8, 8, 1, learningRate=0.0000002, seed=0)
ffnn.setTrainingData(testInputs, testOutputs)

# Train the neural network
iterations = 5000
print('Training the network for {} iterations...'.format(iterations))
ffnn.train(iterations, graph=True, showOutput=False, showWeights=False)

# Validate
validationSamples = 10**3

correctStealNoFakeOut = 0
for i in range(validationSamples):
    seq = bodyCodes.bodyCodeSequenceToIntCode(bodyCodes.randomStealWithoutFakeOut())
    isSteal = True if ffnn.forwardPropagation(seq) > 0.5 else False
    if isSteal:
        correctStealNoFakeOut += 1
stealNoFakeOutPercent = correctStealNoFakeOut / validationSamples

correctStealFakeOut = 0
for i in range(validationSamples):
    seq = bodyCodes.bodyCodeSequenceToIntCode(bodyCodes.randomStealWithFakeOut())
    isSteal = True if ffnn.forwardPropagation(seq) > 0.5 else False
    if not isSteal:
        correctStealFakeOut += 1
stealFakeOutPercent = correctStealFakeOut / validationSamples

correctNotSteal = 0
for i in range(validationSamples):
    seq = bodyCodes.bodyCodeSequenceToIntCode(bodyCodes.randomNoSteal())
    isSteal = True if ffnn.forwardPropagation(seq) > 0.5 else False
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

np.seterr(all='raise')

# Return max
'''
inputs = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [1, -1], [1, 0], [1, 1]])
target = np.array([[-1, 0, 1 , 0, 0, 1, 1, 1]]).T
'''

# Select max of 2
'''
inputs = np.array([[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [1, -1], [1, 0], [1, 1]])
target = np.array([[ 1,  1], [ 0, 1], [ 0, 1], [1,  0], [1, 1], [1,  0], [1, 0], [1, 1]])
'''

# Select max of 3

inputs = np.array([[random.uniform(-1, 1), random.uniform(-1, 1)]
                    for i in range(10**3)])
target = np.array([[1 if v[0]==max(v) else -1,1 if v[1]==max(v) else -1,1 if v[0]*v[1]<0 else -1]
                    for v in inputs])


'''
# Outside of circle
inputs = np.array([[random.uniform(-1, 1), random.uniform(-1, 1)]  
                    for i in range(10**3)])
target = np.array([[1 if (math.sqrt((v[0]-0.5)*(v[0]-0.5)+v[1]*v[1]) < 0.5**2) else -1] for v in inputs])
'''

ffnn = nn.SimpleFFNN(len(inputs[0]), 8, 8, 8, 8, 8, 8, len(target[0]), learningRate=0.000005, seed=0)
ffnn.setTrainingData(inputs, target)
ffnn.train(10**4, graph=True, showOutput=True, showWeights=False)
print('LOSS: ', nn._meanSquared(ffnn.forwardPropagation(inputs) - target))

for i in range(min(15, len(inputs))):
    print(inputs[i], ' : ', ffnn.forwardPropagation(inputs[i]), ' should be ', target[i])

#for pair in [[-1, -1], [-1, 0], [-1, 1], [0, -1], [0, 0], [0, 1], [1, -1], [1, 0], [1, 1]]:
#    print('max(', pair, ') = ', ffnn.forwardPropagation(pair))