import numpy as np
import pandas as pd
from fairlearn.datasets import fetch_diabetes_hospital
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.utils import shuffle

from src.metric import showgraph

def load_adult_overpre(random_seed):
    path = '/data/cs_hlian001/fairness_project/data/adult/adult.data'
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship',
              'race', 'sex', 'capital-gain',
              'capital-loss', 'hours-per-week', 'native-country', 'salary']
    df = pd.read_csv(path, index_col=False, skipinitialspace=True, header=None, names=header)
    df = df.replace('?', np.nan)
    df[pd.isnull(df).any(axis=1)].shape
    # 2.1.2. Drop rows containing nan
    df.dropna(inplace=True)
    #uniq = df['occupation'].unique().tolist()
    #occupation = uniq
    df.drop('education-num', axis=1, inplace=True)
    #df.drop('occupation', axis=1, inplace=True)
    categorical_columns = ['workclass', 'education', 'marital-status', 'relationship','race', 'sex',
                           'native-country']
    label_column = ['salary']

    # 2.2.2. convert categoricals to numerical
    def show_unique_values(columns):
        for column in columns:
            uniq = df[column].unique().tolist()
    show_unique_values(categorical_columns)
    show_unique_values(label_column)

    # 2.2.2.1. convert to int
    def convert_to_int(columns):
        for column in columns:
            unique_values = df[column].unique().tolist()
            dic = {}
            for indx, val in enumerate(unique_values):
                dic[val] = indx
            df[column] = df[column].map(dic).astype(int)


    convert_to_int(label_column)
    show_unique_values(label_column)


    # 2.2.2.2. convert to one-hot (good one)
    def convert_to_onehot(data, columns):
        dummies = pd.get_dummies(data[columns])
        data = data.drop(columns, axis=1)
        data = pd.concat([data, dummies], axis=1)
        return data
    df = convert_to_onehot(df, categorical_columns)

    df['occupation'] = pd.factorize(df['occupation'])[0]

    normalize_columns = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    def show_values(columns):
        for column in columns:
            max_val = df[column].max()
            min_val = df[column].min()
            mean_val = df[column].mean()
            var_val = df[column].var()
    show_values(normalize_columns)

    def normalize(columns):
        scaler = preprocessing.StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
    normalize(normalize_columns)
    show_values(normalize_columns)

    data = df.drop('salary', axis=1)
    label = df['salary']
    occupation = df['occupation'].values

    def get_sex(data):
        gender = []
        for index, row in data.iterrows():
            sex_Male = row['sex_Male']
            if sex_Male:
                gender.append(1)
            else:
                gender.append(0)
        return gender
    sex = get_sex(df)
    data = data.values
    label = label.values

    #dataset = np.delete(dataset, 6, axis=1)
    data, label, sex, occupation = shuffle(data, label, sex, occupation, random_state=random_seed)
    return data, label, sex, occupation
def load_adult_overpre_test(random_seed):
    path = '/data/cs_hlian001/fairness_project/data/adult/adult.test'
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship',
              'race', 'sex', 'capital-gain',
              'capital-loss', 'hours-per-week', 'native-country', 'salary']
    df = pd.read_csv(path, index_col=False, skipinitialspace=True, header=None, names=header)
    df = df.replace('?', np.nan)
    df[pd.isnull(df).any(axis=1)].shape
    # 2.1.2. Drop rows containing nan
    df.dropna(inplace=True)

    df.drop('education-num', axis=1, inplace=True)
    categorical_columns = ['workclass', 'education', 'marital-status', 'relationship','race', 'sex',
                           'native-country']
    label_column = ['salary']

    # 2.2.2. convert categoricals to numerical
    def show_unique_values(columns):
        for column in columns:
            uniq = df[column].unique().tolist()
    show_unique_values(categorical_columns)
    show_unique_values(label_column)

    # 2.2.2.1. convert to int
    def convert_to_int(columns):
        for column in columns:
            unique_values = df[column].unique().tolist()
            dic = {}
            for indx, val in enumerate(unique_values):
                dic[val] = indx
            df[column] = df[column].map(dic).astype(int)


    convert_to_int(label_column)
    show_unique_values(label_column)


    # 2.2.2.2. convert to one-hot (good one)
    def convert_to_onehot(data, columns):
        dummies = pd.get_dummies(data[columns])
        data = data.drop(columns, axis=1)
        data = pd.concat([data, dummies], axis=1)
        return data
    df = convert_to_onehot(df, categorical_columns)

    df['occupation'] = pd.factorize(df['occupation'])[0]

    normalize_columns = ['age', 'fnlwgt', 'capital-gain', 'capital-loss', 'hours-per-week']
    def show_values(columns):
        for column in columns:
            max_val = df[column].max()
            min_val = df[column].min()
            mean_val = df[column].mean()
            var_val = df[column].var()
    show_values(normalize_columns)

    def normalize(columns):
        scaler = preprocessing.StandardScaler()
        df[columns] = scaler.fit_transform(df[columns])
    normalize(normalize_columns)
    show_values(normalize_columns)

    data = df.drop('salary', axis=1)
    label = df['salary']
    occupation = df['occupation'].values

    def get_sex(data):
        gender = []
        for index, row in data.iterrows():
            sex_Male = row['sex_Male']
            if sex_Male:
                gender.append(1)
            else:
                gender.append(0)
        return gender
    sex = get_sex(df)
    data = data.values
    label = label.values

    #dataset = np.delete(dataset, 6, axis=1)
    data, label, sex, occupation = shuffle(data, label, sex, occupation, random_state=random_seed)
    return data, label, sex, occupation

def load_adult(random_seed):
    path = './data/adult/adult.data'
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship',
              'race', 'sex', 'capital-gain',
              'capital-loss', 'hours-per-week', 'native-country', 'salary']
    df = pd.read_csv(path, index_col=False, skipinitialspace=True, header=None, names=header)
    df = df.replace('?', np.nan)
    df[pd.isnull(df).any(axis=1)].shape
    # 2.1.2. Drop rows containing nan
    df.dropna(inplace=True)
    string_cols = df.select_dtypes(include=['object']).columns
    code = {'Adm-clerical': 0, 'Farming-fishing': 4, 'Tech-support': 1, 'Craft-repair': 2, 'Transport-moving': 8,
            'Machine-op-inspct': 6, 'Exec-managerial': 3, 'Prof-specialty': 9, 'Protective-serv': 10,
            'Priv-house-serv': 13, 'Other-service': 7, 'Armed-Forces': 12, 'Handlers-cleaners': 5, 'Sales': 11}
    df["occupation"] = df["occupation"].map(code)
    df[string_cols] = df[string_cols].apply(lambda x: x.astype('category').cat.codes)
    dataset = df.values
    sex = dataset[:, 9]
    workclass = dataset[:, 6]
    #dataset = np.delete(dataset, 9, axis=1)
    dataset = np.delete(dataset, 6, axis=1)
    data = dataset[:, :-1]
    label = dataset[:, -1]
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data, label, sex, occupation = shuffle(data, label, sex, workclass, random_state=random_seed)
    return data, label, sex, occupation

def load_adult_test(random_seed):
    path = './data/adult/adult.test'
    header = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation',
              'relationship',
              'race', 'sex', 'capital-gain',
              'capital-loss', 'hours-per-week', 'native-country', 'salary']
    df = pd.read_csv(path, index_col=False, skipinitialspace=True, header=None, names=header)
    df = df.replace('?', np.nan)
    df[pd.isnull(df).any(axis=1)].shape
    # 2.1.2. Drop rows containing nan
    df.dropna(inplace=True)
    string_cols = df.select_dtypes(include=['object']).columns
    code = {'Adm-clerical': 0, 'Farming-fishing': 4, 'Tech-support': 1, 'Craft-repair': 2, 'Transport-moving':8, 'Machine-op-inspct': 6, 'Exec-managerial': 3, 'Prof-specialty': 9, 'Protective-serv': 10, 'Priv-house-serv': 13, 'Other-service': 7, 'Armed-Forces': 12, 'Handlers-cleaners': 5, 'Sales': 11}
    df["occupation"] = df["occupation"].map(code)
    df[string_cols] = df[string_cols].apply(lambda x: x.astype('category').cat.codes)
    dataset = df.values
    sex = dataset[:, 9]
    workclass = dataset[:, 6]
    #dataset = np.delete(dataset, 9, axis=1)
    dataset = np.delete(dataset, 6, axis=1)
    data = dataset[:, :-1]
    label = dataset[:, -1]
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data, label, sex, occupation = shuffle(data, label, sex, workclass, random_state=random_seed)
    return data, label, sex, occupation

def load_kdd_sensus_income(random_seed):
    path = './data/KDD_census_income/census-income.data'
    dataframe = pd.read_csv(path, header=None)
    dataframe = dataframe.drop(columns=[27, 28, 29, 31])
    dataframe = dataframe.loc[~(dataframe == '?').any(axis=1)]
    string_cols = dataframe.select_dtypes(include=['object']).columns
    dataframe[string_cols] = dataframe[string_cols].apply(lambda x: x.astype('category').cat.codes)
    dataset = dataframe.values
    sex = dataset[:, 12]
    occupation = dataset[:, 1]
    #dataset = np.delete(dataset, 9, axis=1)
    dataset = np.delete(dataset, 1, axis=1)
    data = dataset[:, :-1]
    label = dataset[:, -1]
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data, label, sex, occupation = shuffle(data, label, sex, occupation, random_state=random_seed)
    return data, label, sex, occupation

def load_kdd_sensus_income_test(random_seed):
    path = './data/KDD_census_income/census-income.test'
    dataframe = pd.read_csv(path, header=None)
    dataframe = dataframe.drop(columns=[27, 28, 29, 31])
    dataframe = dataframe.loc[~(dataframe == '?').any(axis=1)]
    string_cols = dataframe.select_dtypes(include=['object']).columns
    dataframe[string_cols] = dataframe[string_cols].apply(lambda x: x.astype('category').cat.codes)
    dataset = dataframe.values
    sex = dataset[:, 12]
    occupation = dataset[:, 1]
    #dataset = np.delete(dataset, 9, axis=1)
    dataset = np.delete(dataset, 1, axis=1)
    data = dataset[:, :-1]
    label = dataset[:, -1]
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    data, label, sex, occupation = shuffle(data, label, sex, occupation, random_state=random_seed)
    return data, label, sex, occupation

def load_dutch_sensus_income(random_seed):
    path = './data/dutch_census/dutch_census_2001.arff.txt'
    dataframe = pd.read_csv(path, header=None)
    string_cols = dataframe.select_dtypes(include=['object']).columns
    dataframe[string_cols] = dataframe[string_cols].apply(lambda x: x.astype('category').cat.codes)
    dataset = dataframe.values
    data = dataset[:, :-1]
    label = dataset[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=random_seed)
    sex = X_train[:, 0]
    sex[sex == 2] = 0
    education = X_train[:,1]
    education = np.array(list(map(lambda x: x - 4, education)))
    #dataset = np.delete(dataset, 9, axis=1)
    X_train = np.delete(X_train, 1, axis=1)
    y_train = np.array(list(map(lambda x: 1 - x, y_train)))

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    return X_train, y_train, sex, education

def load_dutch_sensus_income_test(random_seed):
    path = './data/dutch_census/dutch_census_2001.arff.txt'
    dataframe = pd.read_csv(path, header=None)
    string_cols = dataframe.select_dtypes(include=['object']).columns
    dataframe[string_cols] = dataframe[string_cols].apply(lambda x: x.astype('category').cat.codes)
    dataset = dataframe.values
    data = dataset[:, :-1]
    label = dataset[:, -1]
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=random_seed)
    sex = X_test[:, 0]
    sex[sex == 2] = 0
    education = X_test[:, 1]
    education = np.array(list(map(lambda x: x - 4, education)))
    # dataset = np.delete(dataset, 9, axis=1)
    X_test = np.delete(X_test, 1, axis=1)
    y_test = np.array(list(map(lambda x: 1 - x, y_test)))

    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    return X_test, y_test, sex, education


def load_bank(random_seed):
    path = './data/bank/bank-full.csv'
    dataframe = pd.read_csv(path, sep=';')
    string_cols = dataframe.select_dtypes(include=['object']).columns
    dataframe[string_cols] = dataframe[string_cols].apply(lambda x: x.astype('category').cat.codes)
    dataset = dataframe.values
    label = dataset[:, -1]
    data = dataset[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=random_seed)
    marial = X_train[:, 2]
    marial[marial == 2] = 0
    marial = np.array(list(map(lambda x: 1 - x, marial)))
    occupation = X_train[:, 1]
    #marial = np.array([1 - value for value in marial])
    #dataset = np.delete(dataset, 9, axis=1)
    X_train = np.delete(X_train, 10, axis=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    return X_train, y_train, marial, occupation

def load_bank_test(random_seed):
    path = './data/bank/bank-full.csv'
    dataframe = pd.read_csv(path, sep=';')
    string_cols = dataframe.select_dtypes(include=['object']).columns
    dataframe[string_cols] = dataframe[string_cols].apply(lambda x: x.astype('category').cat.codes)
    dataset = dataframe.values
    label = dataset[:, -1]
    data = dataset[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=random_seed)
    marial = X_test[:, 2]
    marial[marial == 2] = 0
    marial = np.array(list(map(lambda x: 1 - x, marial)))
    occupation = X_test[:, 1]
    #marial = np.array([1 - value for value in marial])
    # dataset = np.delete(dataset, 9, axis=1)
    X_test = np.delete(X_test, 10, axis=1)
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    return X_test, y_test, marial, occupation

def load_diabetes(random_seed):
    path = './data/diabetes/dataset_diabetes/diabetic_preprocessed.csv'
    dataframe = pd.read_csv(path, sep=',')
    string_cols = dataframe.select_dtypes(include=['object']).columns
    dataframe[string_cols] = dataframe[string_cols].apply(lambda x: x.astype('category').cat.codes)
    dataframe = dataframe.astype(int)
    dataframe = dataframe.iloc[:, :-1]
    dataset = dataframe.values
    label = dataset[:, -1]
    data = dataset[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=random_seed)
    gender = X_train[:, 1]
    gender[gender == 2] = 1
    gender = np.array(list(map(lambda x: 1 - x, gender)))
    occupation = X_train[:, 2]
    X_train = np.delete(X_train, 2, axis=1)
    X_train = np.delete(X_train, -1, axis=1)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    return X_train, y_train, gender, occupation

def load_diabetes_test(random_seed):
    path = './data/diabetes/dataset_diabetes/diabetic_preprocessed.csv'
    dataframe = pd.read_csv(path, sep=',')
    string_cols = dataframe.select_dtypes(include=['object']).columns
    dataframe[string_cols] = dataframe[string_cols].apply(lambda x: x.astype('category').cat.codes)
    dataframe = dataframe.astype(int)
    dataframe = dataframe.iloc[:, :-1]
    dataset = dataframe.values
    label = dataset[:, -1]
    data = dataset[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=random_seed)
    gender = X_test[:, 1]
    gender[gender == 2] = 1
    gender = np.array(list(map(lambda x: 1 - x, gender)))
    occupation = X_test[:, 2]
    X_test = np.delete(X_test, 2, axis=1)
    X_test = np.delete(X_test, -1, axis=1)
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)

    return X_test, y_test, gender, occupation

def load_lawschool(random_seed):
    path = './data/lawschool/bar_pass_prediction.csv'
    dataframe = pd.read_csv(path)
    dataframe = dataframe.dropna()
    string_cols = dataframe.select_dtypes(include=['object']).columns
    dataframe[string_cols] = dataframe[string_cols].apply(lambda x: x.astype('category').cat.codes)
    dataframe['indxgrp2'] = pd.factorize(dataframe['indxgrp2'])[0]
    label = dataframe['pass_bar'].values
    #label = np.array(list(map(lambda x: 1 - x, label)))
    dataframe = dataframe.drop('pass_bar', axis = 1)
    dataset = dataframe.values
    data = dataset[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=random_seed)
    race = X_train[:, 4]
    race = np.where(race != 7, 0, 1)
    tier = X_train[:,19]
    tier = np.array(list(map(lambda x: x - 1, tier)))
    X_train = np.delete(X_train, 19, axis=1)
    #y_train = np.array(list(map(lambda x: 1 - x, y_train)))
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
    #label = np.array(list(map(lambda x: 1 - x, label)))
    dataframe = dataframe.drop('pass_bar', axis=1)
    dataset = dataframe.values
    data = dataset[:, :-1]
    X_train, X_test, y_train, y_test = train_test_split(data, label, test_size=0.3, random_state=random_seed)
    race = X_test[:, 4]
    race = np.where(race != 7, 0, 1)
    tier = X_test[:, 19]
    tier = np.array(list(map(lambda x: x - 1, tier)))
    X_test = np.delete(X_test, 19, axis=1)
    # y_train = np.array(list(map(lambda x: 1 - x, y_train)))
    scaler = StandardScaler()
    X_test = scaler.fit_transform(X_test)
    return X_test, y_test, race, tier

