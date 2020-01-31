import numpy as np
from abc import ABC, abstractmethod

class ActivationFunction(ABC):

    @abstractmethod
    def compute(self, x):
        pass

    @abstractmethod
    def computeDer(self, x):
        pass

class SigmoidAF(ActivationFunction):

    def compute(self, x):
        return 1.0 / (1.0 + np.exp(-x))

    def computeDer(self, x):
        return self.compute(x) * (1.0 - self.compute(x))

class AtanAF(ActivationFunction):

    def compute(self, x):
        return np.arctan(x)

    def computeDer(self, x):
        return 1.0 / (x**2 + 1.0)


class ReluAF(ActivationFunction):

    def compute(self, x):
        z = np.copy(x)
        return z * (z > 0)

    def computeDer(self, x):
        z = np.copy(x)
        return 1.0 * (z > 0)

class LeakyReluAF(ActivationFunction):

    def compute(self, x):
        z = np.copy(x)
        return np.where(z > 0, z, z * 0.01)

    def computeDer(self, x):
        z = np.copy(x)
        return np.nan_to_num(np.where(z > 0, 1, 0.01))