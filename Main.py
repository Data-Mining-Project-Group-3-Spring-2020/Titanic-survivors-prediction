from Datasets import Datasets as dataset
from NN import NN
import os, glob, numpy as np

if __name__ == "__main__":
    data = dataset()
    data.read_data()
    training, label = data.get_titanic_training_data()
    testing, label_test = data.get_titanic_test_data()

    print(training.shape)
    print(testing.shape)

    ai = NN()

    # Training
    ai.create_model()
    ai.train_model(training, label, 500)
    ai.model.summary()

    # Testing
    # ai.load_model()
    ai.test_model(testing, label_test)
