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
        self.lusitania_path = os.path.join(os.getcwd(), "passengermanifest.xlsx")

        # get data from files
        self.read_data()

        # get input and expected output
        self.sort_data()

    def read_data(self):
        # extract data from csv and xls file to pandas data frames
        self.titanic_train_df = pd.read_csv(self.titanic_path_train).fillna(0)
        self.titanic_test_df = pd.read_csv(self.titanic_path_test).fillna(0)
        self.titanic_test_out_df = pd.read_csv(self.titanic_path_test_out).fillna(0)
        self.lusitania_df = pd.read_excel(self.lusitania_path).fillna(0)

    def sort_data(self):
        # ------------------------------------ titanic ------------------------------------------------------
        # spit out from the data frame
        # ------------------------------------ training ------------------------------------------
        # titanic_train_df_sex = self.titanic_train_df['Sex']
        # titanic_train_df_embarked = self.titanic_train_df['Embarked']
        titanic_train_sex = pd.DataFrame(columns=['Sex'])
        # titanic_train_embarked = pd.DataFrame(columns=['Embarked'])
        self.titanic_train = self.titanic_train_df.drop(columns=['PassengerId', 'Survived', 'Name', 'Ticket', 'Cabin', 'Sex', 'Embarked', "Fare"])
         
        for index, item in self.titanic_train_df.iterrows():
            if item['Sex'] == 'male':
                titanic_train_sex = titanic_train_sex.append({'Sex': 1}, ignore_index=True)
            else:
                titanic_train_sex = titanic_train_sex.append({'Sex': 0}, ignore_index=True)

            '''if item['Embarked'] == 'S':
                titanic_train_embarked = titanic_train_embarked.append({'Embarked': 0}, ignore_index=True)
            elif item['Embarked'] == 'C':
                titanic_train_embarked = titanic_train_embarked.append({'Embarked': 1}, ignore_index=True)
            else:
                titanic_train_embarked = titanic_train_embarked.append({'Embarked': 2}, ignore_index=True)'''
       

        self.titanic_train_out = self.titanic_train_df['Survived']
        self.titanic_train = pd.concat([self.titanic_train, titanic_train_sex], axis=1)  #  titanic_train_embarked], axis=1)

        # --------------------------------------- test -----------------------------------------------
        # titanic_test_df_sex = self.titanic_test_df["Sex"]
        # titanic_test_df_embarked = self.titanic_test_df['Embarked']
        titanic_test_sex = pd.DataFrame(columns=['Sex'])
        # titanic_test_embarked = pd.DataFrame(columns=['Embarked'])
        # print(self.titanic_test_df.columns)
        self.titanic_test = self.titanic_test_df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin', 'Sex', 'Embarked', "Fare"])

        for index, item in self.titanic_test_df.iterrows():
            if item['Sex'] == 'male':
                titanic_test_sex = titanic_test_sex.append({'Sex': 1}, ignore_index=True)
            else:
                titanic_test_sex = titanic_test_sex.append({'Sex': 0}, ignore_index=True)

            '''if item['Embarked'] == 'S':
                titanic_test_embarked = titanic_test_embarked.append({'Embarked': 0}, ignore_index=True)
            elif item['Embarked'] == 'C':
                titanic_test_embarked = titanic_test_embarked.append({'Embarked': 1}, ignore_index=True)
            else:
                titanic_test_embarked = titanic_test_embarked.append({'Embarked': 2}, ignore_index=True)'''

        self.titanic_test_out = self.titanic_test_out_df['Survived']
        self.titanic_test = pd.concat([self.titanic_test, titanic_test_sex], axis=1) #, titanic_test_embarked], axis=1)
        # ------------------------------------------ titanic -----------------------------------------------

        # ------------------------------------------ lusitania ---------------------------------------------
        # print(self.lusitania_df.columns)
        #self.lusitania_test = self.lusitania_df.drop(columns=[])
        #-- Using the convention above, added similar for lusitania
        lusitania_test_df_sex = self.lusitania_df['Sex']
        lusitania_test_sex = pd.DataFrame(columns=['Sex'])
        lusitania_test_out = pd.DataFrame(columns=['Survived'])
        self.lusitania_test = self.lusitania_df.drop(columns=['PassengerId', 'Survived', 'Sex'])
               #-- Changes sex from male/female to 1/0 for Lusitania
        for index, item in self.lusitania_df.iterrows():
            if item['Sex'] == 'male' or item['Sex'] == 'Male':
                lusitania_test_sex = lusitania_test_sex.append({'Sex': 1}, ignore_index=True)
            else:
                lusitania_test_sex = lusitania_test_sex.append({'Sex': 0}, ignore_index=True)

            if item['Survived'] == 'Saved':
                lusitania_test_out = lusitania_test_out.append({'Survived': 1}, ignore_index=True)
            else:
                lusitania_test_out = lusitania_test_out.append({"Survived": 0}, ignore_index=True)

        self.lusitania_test_out = lusitania_test_out
        self.lusitania_test = pd.concat([self.lusitania_test, lusitania_test_sex], axis=1)
        # print(self.lusitania_test)
        # print(self.lusitania_test_out)
        # ------------------------------------------ lusitania ---------------------------------------------


    def get_titanic_training_data(self):
        # return input and expected values
        print(self.titanic_train.columns)
        return self.titanic_train.to_numpy(dtype=np.float32()), self.titanic_train_out.to_numpy(dtype=np.float32())

    def get_titanic_test_data(self):
        # return input and expected values
        # print(self.titanic_test.columns)
        return self.titanic_test.to_numpy(dtype=np.float32()), self.titanic_test_out.to_numpy(dtype=np.float32())

    def get_lusitania(self):
        print(self.lusitania_test.columns)
        return self.lusitania_test.to_numpy(dtype=np.float32()), self.lusitania_test_out.to_numpy(dtype=np.float32())


if __name__ == "__main__":
    d = Datasets()
    d.read_data()
    #d.sort_data()
    x, y = d.get_titanic_training_data()
    print(x)
    print(y)

