#!/usr/bin/env python3

import os
import tarfile
from six.moves import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import Imputer

DOWNLOAD_ROOT = 'https://raw.githubusercontent.com/ageron/handson-ml/master/'
HOUSING_PATH = os.path.join('datasets', 'housing')
HOUSING_URL = DOWNLOAD_ROOT + 'datasets/housing/housing.tgz'


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, 'housing.tgz')
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, 'housing.csv')
    return pd.read_csv(csv_path)


def split_train_test(data, test_ratiao):
    '''
    simple to split the dataset, but each time
    the result would be different
    '''
    shuffled_indeces = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratiao)
    test_indices = shuffled_indeces[:test_set_size]
    train_indices = shuffled_indeces[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]


if __name__ == '__main__':
    #  fetch_housing_data()
    housing = load_housing_data()
    #  housing.hist(bins=50, figsize=(20, 15))
    #  plt.show()
    # Method1:  split the housing dataset into train and test
    train_set, test_set = split_train_test(housing, 0.2)
    print(len(train_set), 'train + ', len(test_set), 'test')
    '''
    Method2: split dataset
    create an income category attribute by dividing the median income by 1.5
    the purpose is split instance into different straums
    '''
    housing['income_cat'] = np.ceil(housing['median_income'] / 1.5)
    housing['income_cat'].where(housing['income_cat'] < 5, 5.0, inplace=True)
    #  housing['income_cat'].hist()
    #  plt.show()
    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing['income_cat']):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    #  print(strat_test_set['income_cat'].value_counts() / len(strat_test_set))
    # remove `income_cat`
    for set_ in (strat_train_set, strat_test_set):
        set_.drop('income_cat', axis=1, inplace=True)
    housing = strat_train_set
    housing.plot(
        kind='scatter',
        x='longitude',
        y='latitude',
        alpha=0.4,
        s=housing['population'] / 100,
        label='population',
        figsize=(10, 7),
        c='median_house_value',
        cmap=plt.get_cmap('jet'),
        colorbar=True)
    #  plt.show()
    # compute the stadard correlation coefficient
    corr_matrix = housing.corr()
    print(corr_matrix['median_house_value'].sort_values(ascending=False))
    # compute correlation between attributes with `scatter_matrix`
    attributes = [
        'median_house_value', 'median_income', 'total_rooms',
        'housing_median_age'
    ]
    pd.plotting.scatter_matrix(housing[attributes], figsize=(10, 6))
    #  plt.show()
    housing.plot(
        kind='scatter', x='median_income', y='median_house_value', alpha=0.1)
    #  plt.show()
    housing['rooms_pre_household'] = housing['total_rooms'] / \
        housing['households']
    housing['bedrooms_per_room'] = housing['total_bedrooms'] / \
        housing['total_rooms']
    housing['population_per_room'] = housing['population'] / \
        housing['households']
    corr_matrix = housing.corr()
    print(corr_matrix['median_house_value'].sort_values(ascending=False))
    # prepare the data: separate the predictors and the labels
    housing = strat_train_set.drop('median_house_value', axis=1)
    housing_labels = strat_train_set['median_house_value'].copy()
    # data cleaning: measure the missing value
    inputer = Imputer(strategy='median')  # replace by median value
    housing_num = housing.drop('ocean_proximity', axis=1)
    inputer.fit(housing_num)
