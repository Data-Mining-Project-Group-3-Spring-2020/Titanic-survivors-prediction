import tensorflow as tf
import pandas as pd
import numpy as np
import os

class NN:
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.checkpoint_path = "Titanic10_4/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

    def create_model(self, layer=[10, 4]):
        # Add hidden layers
        for amount in layer:
            self.model.add(tf.keras.layers.Dense(amount, activation=tf.nn.leaky_relu))
        self.model.add(tf.kers.layers.Dense(1, activation=tf.nn.leaky_relu))

        # build model
        self.model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    def train_model(self, x, y, epoch=100):
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, save_weights_only=True, verbose=1)

        # train model
        self.model.fit(x, y, epochs=epoch, callbacks=[cp_callback])

    def load_model(self):
        # remember to match layer amounts to the save model
        self.create_model([10, 4])

        # load trained weights
        self.model.load_weights(self.checkpoint_path)

        # print model specs
        self.model.summary()

    def test_model(self, x, y):
        self.model.evaluate(x, y)

    def run_model(self, x):  # x => numpy.ndarrary
        return np.argmax(self.model.predict(x))
