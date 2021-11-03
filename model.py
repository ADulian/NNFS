from layers import Input
import numpy as np

class Model:
    def __init__(self):
        # Create list of network objects
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer, accuracy):
        self.loss = loss
        self.optimizer = optimizer
        self.accuracy = accuracy

    def train(self, X, y, *, epochs=1, print_every=1):

        self.accuracy.init(y)

        for epoch in range(1, epochs+1):
            # Forward pass
            output = self.forward(X)

            # Loss
            data_loss, regularization_loss = self.loss.calculate(output, y)
            loss = data_loss + regularization_loss

            # Get predictions and calculate an accuracy
            predictions = self.output_layer_activation.predictions(output)
            accuracy = self.accuracy.calculate(predictions, y)

            # Temp
            print(output)
            exit()

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

    def forward(self, X):

        # Call forward on input layer to set it's output accordingly
        self.input_layer.forward(X)

        # Forward over network's layers
        for layer in self.layers:
            layer.forward(layer.prev.output)

        # Layer is now the last object from the list, return its output
        return layer.output
