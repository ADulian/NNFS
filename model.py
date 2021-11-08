from layers import Input
from losses import CategoricalCrossEntropy, Softmax_CategoricalCrossentropy
from activations import SoftMax
import numpy as np

class Model:
    def __init__(self):
        # Create list of network objects
        self.layers = []

        self.softmax_classifier_output = None

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def train(self, X, y, *, epochs=1, batch_size=None, print_every=1, validation_data=None):

        self.accuracy.init(y)

        # Default value if batch size is not set
        train_set = 1

        # Validation data defauly step size
        if validation_data is not None:
            validation_steps = 1

            # Data
            X_val, y_val = validation_data

        # Calculate number of steps
        if batch_size is not None:
            train_steps = len(X) // batch_size

            if train_steps * batch_size < len(X):
                train_steps += 1

            if validation_data is not None:
                validation_steps = len(X_val) // batch_size
                if validation_steps * batch_size < len(X_val):
                    validation_steps += 1

        for epoch in range(1, epochs+1):
            print(f'\nepoch: {epoch}')

            # Reset accumulated values in loss and acc objects
            self.loss.new_pass()
            self.accuracy.new_pass()

            # Iterate over steps
            for step in range(train_steps):
                if batch_size is None:
                    batch_X = X
                    batch_y = y
                else:
                    batch_X = X[step*batch_size : (step+1)*batch_size]
                    batch_y = y[step*batch_size : (step+1)*batch_size]

                # Forward pass
                output = self.forward(batch_X, training=True)

                # Loss
                data_loss, regularization_loss = self.loss.calculate(output, batch_y, include_regularization=True)
                loss = data_loss + regularization_loss

                # Get predictions and calculate an accuracy
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, batch_y)

                # Backprop
                self.backward(output, batch_y)

                # Optimize
                self.optimizer.pre_update_params()
                for layer in self.trainable_layers:
                    self.optimizer.update_params(layer)
                self.optimizer.post_update_params()

                if not step % print_every or step == train_steps - 1:
                    print(f'step: {step}, ' +
                          f'acc: {accuracy:.3f}, ' +
                          f'loss: {loss:.4f}, ' +
                          f'data_loss: {data_loss:.4f}, ' +
                          f'reg_loss: {regularization_loss:.4f}, ' +
                          f'lr: {self.optimizer.current_learning_rate}')

            # Print stats at the end of each epochs
            print()
            epoch_data_loss, epoch_regularization_loss = self.loss.calculate_accumulated(include_regularization=True)
            epoch_loss = epoch_data_loss + epoch_regularization_loss
            epoch_accuracy = self.accuracy.calculate_accumulated()

            print(f'training ' +
                  f'acc: {epoch_accuracy:.3f}, ' +
                  f'loss: {epoch_loss:.4f}, ' +
                  f'data_loss: {epoch_data_loss:.4f}, ' +
                  f'reg_loss: {epoch_regularization_loss:.4f}, ' +
                  f'lr: {self.optimizer.current_learning_rate}')
        # Validate
        if validation_data is not None:
            self.loss.new_pass()
            self.accuracy.new_pass()

            for step in range(validation_steps):
                if batch_size is None:
                    batch_X = X_val
                    batch_y = y_val
                else:
                    batch_X = X_val[step*batch_size : (step+1)*batch_size]
                    batch_y = y_val[step*batch_size : (step+1)*batch_size]

                # Forward
                output = self.forward(X_val, training=False)

                # Loss
                loss = self.loss.calculate(output, y_val)

                # Acc
                predictions = self.output_layer_activation.predictions(output)
                accuracy = self.accuracy.calculate(predictions, y_val)

            # Summary
            validation_loss = self.loss.calculate_accumulated()
            validation_accuracy = self.accuracy.calculate_accumulated()
            print(f'\nvalidation: ' + f'acc: {validation_accuracy:.3f}, ' + f'loss: {validation_loss:.3f}')

    def finalize(self):
        # Create and set the input layer
        self.input_layer = Input()

        # Count num layers
        layer_count = len(self.layers)

        # Trainable layers
        self.trainable_layers = []

        # Iterate the objects
        for i in range(layer_count):

            # If it's a first layer the previous layer object is the input layer
            if i == 0:
                self.layers[i].prev = self.input_layer
                self.layers[i].next = self.layers[i + 1]

            # All layers but first and last
            elif i < layer_count - 1:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.layers[i + 1]

            # Last layer's next object is the loss
            else:
                self.layers[i].prev = self.layers[i - 1]
                self.layers[i].next = self.loss
                self.output_layer_activation = self.layers[i]

            # If layer contains an attribute called "weights" it is a trainable layer
            if hasattr(self.layers[i], 'weights'):
                self.trainable_layers.append(self.layers[i])

        # Update loss object with trainable layers
        self.loss.remember_trainable_layers(self.trainable_layers)

        if isinstance(self.layers[-1], SoftMax) and isinstance(self.loss, CategoricalCrossEntropy):
            self.softmax_classifier_output = Softmax_CategoricalCrossentropy()

    def forward(self, X, training):

        # Call forward on input layer to set it's output accordingly
        self.input_layer.forward(X, training)

        # Forward over network's layers
        for layer in self.layers:
            layer.forward(layer.prev.output, training)

        # Layer is now the last object from the list, return its output
        return layer.output

    def backward(self, output, y):
        # Softmax classifier
        if self.softmax_classifier_output is not None:
            self.softmax_classifier_output.backward(output, y)
            self.layers[-1].dinputs = self.softmax_classifier_output.dinputs

            for layer in reversed(self.layers[:-1]):
                layer.backward(layer.next.dinputs)

            return

        # First call backward method on the loss
        self.loss.backward(output, y)

        # Apply backprop to all layers
        for layer in reversed(self.layers):
            layer.backward(layer.next.dinputs)
