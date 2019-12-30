import nn
import numpy as np

ffnn = nn.FeedForwardNN2(2, 16, 8, 1, learningRate=1)
test = np.array([0.5, -1, 0, 1])
print(ffnn._relu(test))
print(ffnn._reluDerivative(test))
print(test)