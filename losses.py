import numpy as np
from activations import SoftMax

class Loss:
    def calculate(self, output, y, *, include_regularization=False):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)

        if not include_regularization:
            return data_loss

        return data_loss, self.regularization_loss()

    def remember_trainable_layers(self, trainable_layers):
        self.trainable_layers = trainable_layers

    def regularization_loss(self):
        # 0 By default
        regularization_loss = 0

        # L1
        for layer in self.trainable_layers:
            if layer.weight_regularizer_l1 > 0:
                regularization_loss += layer.weight_regularizer_l1 * np.sum(np.abs(layer.weights))

            if layer.bias_regularizer_l1 > 0:
                regularization_loss += layer.bias_regularizer_l1 * np.sum(np.abs(layer.biases))

            # L2
            if layer.weight_regularizer_l2 > 0:
                regularization_loss += layer.weight_regularizer_l2 * np.sum(layer.weights * layer.weights)

            if layer.bias_regularizer_l2 > 0:
                regularization_loss += layer.bias_regularizer_l2 * np.sum(layer.biases * layer.biases)

        return regularization_loss

class CategoricalCrossEntropy(Loss):
    def forward(self, y_pred, y_true):
        samples = len(y_pred)

        y_pred_clipped = np.clip(y_pred, 1e-7, 1 - 1e-7)

        if len(y_true.shape) == 1: # Labels
            correct_conf = y_pred_clipped[range(samples), y_true]
        elif len(y_true.shape) == 2: # One-Hot
            correct_conf = np.sum(y_pred_clipped*y_true, axis=1)

        negative_log_likelihoods = -np.log(correct_conf)
        return negative_log_likelihoods

    def backward(self, dvalues, y_true):
        # Number of samples / batches
        samples = len(dvalues)

        # Number of labels in every sample
        labels = len(dvalues[0])

        # If labels are sparse, turn them into one-hot vector
        if len(y_true.shape) == 1:
            y_true = np.eye(labels)[y_true]

        # Calculate gradient
        self.dinputs = -y_true / dvalues

        # Normalize gradient
        self.dinputs = self.dinputs / samples

class Softmax_CategoricalCrossentropy(Loss):
    def __init__(self):
        self.activation = SoftMax()
        self.loss = CategoricalCrossEntropy()

    def forward(self, inputs, y_true):
        # Output layer's activation function
        self.activation.forward(inputs)

        # Outputs
        self.output = self.activation.output

        # Compute and return loss
        return self.loss.calculate(self.output, y_true)

    def backward(self, dvalues, y_true):
        # Number of samples
        samples = len(dvalues)

        # If one-hot encoded turn them into discrete
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis=1)

        self.dinputs = dvalues.copy()

        # Compute gradient
        self.dinputs[range(samples), y_true] -= 1

        # Normalize
        self.dinputs = self.dinputs / samples

class BinaryCrossentropy(Loss):
    def forward(self, y_pred, y_true):
        # Clip data to prevent division by zeros
        # Clip both sides to not drag mean towards any value
        y_pred_clipped = np.clip(y_pred, 1e-7, 1-1e-7)

        # Sample wise loss
        sample_losses = -(y_true * np.log(y_pred_clipped) +
                          (1-y_true) * np.log(1 - y_pred_clipped))
        sample_losses = np.mean(sample_losses, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(y_true)
        outputs = len(dvalues[0])

        # Clip
        clipped_dvalues = np.clip(dvalues, 1e-7, 1-1e-7)

        # Gradient
        self.dinputs = -(y_true / clipped_dvalues -
                         (1-y_true) / (1-clipped_dvalues)) / outputs

        # Normalize gradient
        self.dinputs = self.dinputs / samples

class MeanSquaredError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean((y_true - y_pred)**2, axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = -2*(y_true - dvalues) / outputs

        # Normalize gradients
        self.dinputs = self.dinputs / samples

class MeanAbsoluteError(Loss):
    def forward(self, y_pred, y_true):
        sample_losses = np.mean(np.abs(y_true - y_pred), axis=-1)

        return sample_losses

    def backward(self, dvalues, y_true):
        samples = len(dvalues)
        outputs = len(dvalues[0])

        # Gradient on values
        self.dinputs = np.sign(y_true - dvalues) / outputs

        # Normalize gradients
        self.dinputs = self.dinputs / samples
