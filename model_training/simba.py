from __future__ import print_function
from keras.layers import Dense, Conv2D
from keras.layers import Flatten
from keras.models import Sequential


class SimbaNet:

    def __init__(self, input_shape=(40, 40, 4), output_nodes=1):
        self.input_shape = input_shape
        self.output_nodes = output_nodes
        self.model_name = 'SanityNet'

    def create_model(self):
        model = Sequential(name=self.model_name)
        model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', input_shape=self.input_shape,
                         padding='same', kernel_initializer="he_normal"))
        model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', kernel_initializer="he_normal"))
        model.add(Flatten())
        model.add(Dense(256, activation='relu', kernel_initializer="he_normal"))
        model.add(Dense(256, activation='relu', kernel_initializer="he_normal"))
        model.add(Dense(self.output_nodes, activation='relu', kernel_initializer="he_normal"))
        return model
