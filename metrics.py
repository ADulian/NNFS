import numpy as np

class Accuracy:

    # Compute acc given predictions and ground truth
    def calculate(self, predictions, y):
        # Comaprison results
        comparisons = self.compare(predictions, y)

        # Acc
        acc = np.mean(comparisons)

        # Accumulate
        self.accumulated_sum += np.sum(comparisons)
        self.accumulated_count += len(comparisons)

        return acc

    def calculate_accumulated(self):
        acc = self.accumulated_sum / self.accumulated_count

        return acc

    def new_pass(self):
        self.accumulated_sum = 0
        self.accumulated_count = 0

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

class Accuracy_Categorical(Accuracy):
    def __init__(self, *, binary=False):
        self.binary = binary

    def init(self, y):
        pass

    def compare(self, predictions, y):
        if not self.binary and len(y.shape) == 2:
            y = np.argmax(y, axis=1)

        return predictions == y
