import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import os

class Datasets:
    def __init__(self):
        # files paths
        self.titanic_path_train = os.path.join(os.getcwd(), "train.csv")
        self.titanic_path_test = os.path.join(os.getcwd(), "test.csv")
        self.titanic_path_test_out = os.path.join(os.getcwd(), "gender_submission.csv")
        self.lusitania_path = os.path.join(os.getcwd(), "completemanifest.xls")

        # get data from files
        self.read_data()

        # get input and expected output
        # self.sort_data()

    def read_data(self):
        # extract data from csv and xls file to pandas data frames
        self.titanic_train_df = pd.read_csv(self.titanic_path_train)
        self.titanic_test_df = pd.read_csv(self.titanic_path_test)
        self.titanic_test_out_df = pd.read_csv(self.titanic_path_test_out)
        self.lusitania_df = pd.read_excel(self.lusitania_path)

    def sort_data(self):  # d => panda.dataframe
        # spit out from the data frame
        titanic_train_df_sex = self.titanic_train_df['Sex']
        titanic_train_df_embarked = self.titanic_train_df['Embarked']
        titanic_train_sex = pd.DataFrame(columns=['Sex'])
        titanic_train_embarked = pd.DataFrame(columns=['Embarked'])
        self.titanic_train = self.titanic_train_df.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Sex', 'Embarked'])

        for index, item in self.titanic_train_df.iterrows():
            if item['Sex'] == 'male':
                titanic_train_sex = titanic_train_sex.append({'Sex': 1}, ignore_index=True)
            else:
                titanic_train_sex = titanic_train_sex.append({'Sex': 0}, ignore_index=True)

            if item['Embarked'] == 'S':
                titanic_train_embarked = titanic_train_embarked.append({'Embarked': 0}, ignore_index=True)
            elif item['Embarked'] == 'C':
                titanic_train_embarked = titanic_train_embarked.append({'Embarked': 1}, ignore_index=True)
            else:
                titanic_train_embarked = titanic_train_embarked.append({'Embarked': 2}, ignore_index=True)

        self.titanic_train_out = self.titanic_train_df['Survived']
        self.titanic_train = pd.concat([self.titanic_train, titanic_train_sex, titanic_train_embarked], axis=1)

    def sort(self, d):


    def get_titanic_training_data(self):
        # return input and expected values
        return self.titanic_train.to_numpy(), self.titanic_train_out.to_numpy()

    def get_titanic_test_data(self):
        # return input and expected values
        return self.titanic_test_df.to_numpy(), self.titanic_test_out_df.to_numpy()

d = Datasets()
d.read_data()
d.sort_data()
x, y = d.get_titanic_training_data()
print(x)
print(y)