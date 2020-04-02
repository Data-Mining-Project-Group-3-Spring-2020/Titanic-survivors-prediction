import tensorflow as tf
import pandas as pd
import numpy as np
import os

class NN:
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.checkpoint_path = "Titanic10_4/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)

    def create_model(self):  # layer=[10, 4]
        # Add hidden layers
        #for amount in layer:
        #    print(amount)
        #    self.model.add(tf.keras.layers.Dense(amount, activation=tf.nn.leaky_relu))
        self.model.add(tf.keras.Input(shape=(7,)))
        self.model.add(tf.keras.layers.Dense(15, activation=tf.nn.sigmoid))
        self.model.add(tf.keras.layers.Dense(10, activation=tf.nn.sigmoid))
        self.model.add(tf.keras.layers.Dense(1, activation=tf.nn.sigmoid))

        # build model
        self.model.compile(optimizer='adam', loss=tf.keras.losses.BinaryCrossentropy(), metrics=['accuracy'])

    def train_model(self, x, y, epoch=100):
        # Create a callback that saves the model's weights
        cp_callback = tf.keras.callbacks.ModelCheckpoint(filepath=self.checkpoint_path, save_weights_only=True, verbose=1, save_freq=5000)

        # train model
        self.model.fit(x, y, epochs=epoch, callbacks=[cp_callback], batch_size=10)

    def load_model(self):
        # remember to match layer amounts to the save model
        self.create_model()

        # load trained weights
        self.model.load_weights(self.checkpoint_path)

        self.model.build()

        # print model specs
        self.model.summary()

    def test_model(self, x, y):
        self.model.evaluate(x, y)

    def run_model(self, x):  # x => numpy.ndarrary
        return np.argmax(self.model.predict(x))