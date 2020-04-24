from Datasets import Datasets as dataset
from NN import NN
import os, glob, numpy as np

if __name__ == "__main__":
    data = dataset()
    data.read_data()
    training, label = data.get_titanic_training_data()
    testing, label_test = data.get_titanic_test_data()
    l_test, l_out = data.get_lusitania()

    print("lus", l_test.shape)
    print("titaninc test", testing.shape)
    print("titanic training", training.shape)
    # print(l_out.shape)

    ai = NN()

    # Training
    ai.create_model()
    # ai.load_model()
    ai.train_model(training, label, 500)
    ai.model.summary()

    # Testing
    ai.test_model(testing, label_test)
    ai.test_model(l_test, l_out)
    #print(training.shape)
    #x = np.reshape(training[3], (1, training.shape[1], ))
    #print(x.shape)
    # print(label[3])
    print(ai.run_model(training))
