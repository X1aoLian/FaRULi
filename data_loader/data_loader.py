import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler



def load_lawschool(random_seed):
    path = './data/lawschool/bar_pass_prediction.csv'
    dataframe = pd.read_csv(path)
    dataframe = dataframe.dropna()
    string_cols = dataframe.select_dtypes(include=['object']).columns
    dataframe[string_cols] = dataframe[string_cols].apply(lambda x: x.astype('category').cat.codes)
    dataframe['indxgrp2'] = pd.factorize(dataframe['indxgrp2'])[0]
    label = dataframe['pass_bar'].values
    dataframe = dataframe.drop('pass_bar', axis = 1)
    dataset = dataframe.values
    data = dataset[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=random_seed)
    race = X_train[:, 4]
    race = np.where(race != 7, 0, 1)
    tier = X_train[:,19]
    tier = np.array(list(map(lambda x: x - 1, tier)))
    X_train = np.delete(X_train, 19, axis=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    return X_train, y_train, race, tier

def load_lawschool_test(random_seed):
    path = './data/lawschool/bar_pass_prediction.csv'
    dataframe = pd.read_csv(path)
    dataframe = dataframe.dropna()
    string_cols = dataframe.select_dtypes(include=['object']).columns
    dataframe[string_cols] = dataframe[string_cols].apply(lambda x: x.astype('category').cat.codes)
    dataframe['indxgrp2'] = pd.factorize(dataframe['indxgrp2'])[0]
    label = dataframe['pass_bar'].values
    dataframe = dataframe.drop('pass_bar', axis=1)
    dataset = dataframe.values
    data = dataset[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=random_seed)
    race = X_test[:, 4]
    race = np.where(race != 7, 0, 1)
    tier = X_test[:, 19]
    tier = np.array(list(map(lambda x: x - 1, tier)))
    X_test = np.delete(X_test, 19, axis=1)
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    return X_test, y_test, race, tier

