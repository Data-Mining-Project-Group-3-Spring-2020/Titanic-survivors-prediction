from Datasets import Datasets as dataset
from NN import NN
import os, glob, numpy as np

if __name__ == "__main__":
    data = dataset()
    data.read_data()
    training, label = data.get_titanic_training_data()

    ai = NN()
    ai.create_model()
    ai.train_model(training, label, 1000)
    ai.load_model()
    ai.test_model()
