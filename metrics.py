import numpy as np

class Accuracy:

    # Compute acc given predictions and ground truth
    def calculate(self, predictions, y):
        # Comaprison results
        comparisons = self.compare(predictions, y)

        # Acc
        acc = np.mean(comparisons)

        return acc

class Accuracy_Regression(Accuracy):
    def __init__(self):
        self.precision = None

    # Compute precision value based on passed-in ground truth
    def init(self, y, reinit=False):
        if self.precision is None or reinit:
            self.precision = np.std(y) / 250

    # Compares predictions to the ground truth
    def compare(self, predictions, y):
        return np.absolute(predictions - y) < self.precision
