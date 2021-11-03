from layers import Input
import numpy as np

class Model:
    def __init__(self):
        # Create list of network objects
        self.layers = []

    def add(self, layer):
        self.layers.append(layer)

    def set(self, *, loss, optimizer):
        self.loss = loss
        self.optimizer = optimizer

    def train(self, X, y, *, epochs=1, print_every=1):

        for epoch in range(1, epochs+1):
            # Forward pass
            output = self.forward(X)

            # Temp
            print(output)
            exit()

    def finalize(self):
        # Create and set the input layer
        self.input_layer = Input()

        # Count num layers
        layer_count = len(self.layers)

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

    def forward(self, X):

        # Call forward on input layer to set it's output accordingly
        self.input_layer.forward(X)

        # Forward over network's layers
        for layer in self.layers:
            layer.forward(layer.prev.output)

        # Layer is now the last object from the list, return its output
        return layer.output
