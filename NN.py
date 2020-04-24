import tensorflow as tf
import pandas as pd
import numpy as np
import os

class NN:
    def __init__(self):
        self.model = tf.keras.Sequential()
        self.checkpoint_path = "Titanic10_10/cp.ckpt"
        self.checkpoint_dir = os.path.dirname(self.checkpoint_path)


    def create_model(self):  # layer=[10, 4]
        # Add hidden layers
        self.model.add(tf.keras.Input(shape=(5,)))
        self.model.add(tf.keras.layers.Dense(10, activation=tf.nn.sigmoid))
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

    def run_model(self, input_set):  # x => numpy.ndarrary
        # {pclass: (1,2,3), age:(0-17, 18-25, 26-40, 41-60, 60++), SibSp: (0, 1-2, 3++), Parch:(0, 1-2, 3++), Sex: (Female, Male)}
        dic_srv = {"Pclass": [0, 0, 0], "Age": [0, 0, 0, 0, 0], "SibSp": [0, 0, 0], "Parch": [0, 0, 0], "Sex": [0, 0]}
        dic_total = {"Pclass": [0, 0, 0], "Age": [0, 0, 0, 0, 0], "SibSp": [0, 0, 0], "Parch": [0, 0, 0], "Sex": [0, 0]}
        dic_out = {"Pclass": [0, 0, 0], "Age": [0, 0, 0, 0, 0], "SibSp": [0, 0, 0], "Parch": [0, 0, 0], "Sex": [0, 0]}
        # print("Input set", input_set.shape[0], input_set.shape[1])
        for i in range(input_set.shape[0]):
            #print("Input set "+str(i), input_set[i].shape)
            inp = np.reshape(input_set[i], (1, input_set[i].shape[0],))
            pre = self.model.predict(inp)[0][0] #np.argmax(self.model.predict(x))#
            if pre < 0.5:
                srv = 0
            else:
                srv = 1
            x = inp[0]
            # print(x)
            # print(srv)
            # Pclass
            # print("x[0]", x[0]-1)
            # print(dic_srv["Pclass"][int(x[0]-1)])
            dic_srv["Pclass"][int(x[0]-1)] += srv
            dic_total["Pclass"][int(x[0]-1)] += 1

            # Age ranges
            if x[1] <= 17:
                dic_srv["Age"][0] += srv
                dic_total["Age"][0] += 1
            elif 17 < x[1] <= 25:
                dic_srv["Age"][1] += srv
                dic_total["Age"][1] += 1
            elif 25 < x[1] <= 40:
                dic_srv["Age"][2] += srv
                dic_total["Age"][2] += 1
            elif 40 < x[1] <= 60:
                dic_srv["Age"][3] += srv
                dic_total["Age"][3] += 1
            elif 60 < x[1]:
                dic_srv['Age'][4] += srv
                dic_total["Age"][4] += 1

            # Sibsp ranges
            if x[2] <= 0:
                dic_srv["SibSp"][0] += srv
                dic_total["SibSp"][0] += 1
            elif 1 <= x[2] <= 2:
                dic_srv["SibSp"][1] += srv
                dic_total["SibSp"][1] += 1
            else:
                dic_srv["SibSp"][2] += srv
                dic_total["SibSp"][2] += 1

            # Parch ranges
            if x[3] <= 0:
                dic_srv["Parch"][0] += srv
                dic_total["Parch"][0] += 1
            elif 1 <= x[3] <= 2:
                dic_srv["Parch"][1] += srv
                dic_total["Parch"][1] += 1
            elif 3 <= x[3]:
                dic_srv["Parch"][2] += srv
                dic_total["Parch"][2] += 1

            # Sex ranges
            dic_srv["Sex"][int(x[4])] += srv
            dic_total["Sex"][int(x[4])] += 1

        print(dic_srv)
        print(dic_total)

        dic_out["Pclass"][0] = dic_srv["Pclass"][0] / dic_total["Pclass"][0]
        dic_out["Pclass"][1] = dic_srv["Pclass"][1] / dic_total["Pclass"][1]
        dic_out["Pclass"][2] = dic_srv["Pclass"][2] / dic_total["Pclass"][2]

        dic_out["Age"][0] = dic_srv["Age"][0] / dic_total["Age"][0]
        dic_out["Age"][1] = dic_srv["Age"][1] / dic_total["Age"][1]
        dic_out["Age"][2] = dic_srv["Age"][2] / dic_total["Age"][2]
        dic_out["Age"][3] = dic_srv["Age"][3] / dic_total["Age"][3]
        dic_out["Age"][4] = dic_srv["Age"][4] / dic_total["Age"][4]

        dic_out["SibSp"][0] = dic_srv["SibSp"][0] / dic_total["SibSp"][0]
        dic_out["SibSp"][1] = dic_srv["SibSp"][1] / dic_total["SibSp"][1]
        dic_out["SibSp"][2] = dic_srv["SibSp"][2] / dic_total["SibSp"][2]

        dic_out["Parch"][0] = dic_srv["Parch"][0] / dic_total["Parch"][0]
        dic_out["Parch"][1] = dic_srv["Parch"][1] / dic_total["Parch"][1]
        dic_out["Parch"][2] = dic_srv["Parch"][2] / dic_total["Parch"][2]

        dic_out["Sex"][0] = dic_srv["Sex"][0] / dic_total["Sex"][0]
        dic_out["Sex"][1] = dic_srv["Sex"][1] / dic_total["Sex"][1]

        return dic_out