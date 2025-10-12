import pickle
import torch
import numpy as np
import scipy.io as sio
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit


class dataset4():
    def __init__(self, datadir="../Datasets"):
        self.datadir = datadir
        self.views = ['view1', 'view2', 'view3', 'view4', 'view5']
        self.path_to_data = self.datadir + "/YoutubeFace/view/"
        self.num_class = 31

    def data(self):
        X = {}
        for i, view in enumerate(self.views):
            X[view] = torch.tensor(np.load(open(self.path_to_data + f'v{i}.npy', "rb")), dtype=torch.float)
        y = torch.tensor(np.load(open(self.path_to_data + 'y.npy', "rb"))).squeeze()
        X_train, y_train, X_test, y_test = stratified_split_data(X, y)


        return X_train, y_train, X_test, y_test

class YoutubeFace(dataset4):
    pass

class dataset5():
    def __init__(self, datadir="../Datasets"):
        self.datadir = datadir
        self.views = ['view1', 'view2', 'view3', 'view4', 'view5', 'view6', 'view7']
        self.path_to_data = self.datadir + "/nus_wide/view/"
        self.num_class = 10

    def data(self):
        X = {}
        X['view1'] = torch.tensor(np.load(open(self.path_to_data + 'BoW_int.npy', "rb")), dtype=torch.float)
        X['view2'] = torch.tensor(np.load(open(self.path_to_data + 'Normalized_CH.npy', "rb")), dtype=torch.float)
        X['view3'] = torch.tensor(np.load(open(self.path_to_data + 'Normalized_CM55.npy', "rb")), dtype=torch.float)
        X['view4'] = torch.tensor(np.load(open(self.path_to_data + 'Normalized_CORR.npy', "rb")), dtype=torch.float)
        X['view5'] = torch.tensor(np.load(open(self.path_to_data + 'Normalized_EDH.npy', "rb")), dtype=torch.float)
        X['view6'] = torch.tensor(np.load(open(self.path_to_data + 'Normalized_WT.npy', "rb")), dtype=torch.float)
        X['view7'] = torch.tensor(np.load(open(self.path_to_data + 'tags1k.npy', "rb")), dtype=torch.float)
        y = torch.tensor(np.load(open(self.path_to_data + 'y.npy', "rb"))).squeeze()
        X_train, y_train, X_test, y_test = stratified_split_data(X, y)
        return X_train, y_train, X_test, y_test

class nus_wide(dataset5):
    pass


def stratified_split_data(X, y, test_size=0.2):
    sss = StratifiedShuffleSplit(n_splits=5, test_size=test_size, random_state=1024)
    idx_split = 0
    train_idxs, test_idxs = [], []
    for train_idx, test_idx in sss.split(X['view1'], y):
        train_idxs.append(train_idx)
        test_idxs.append(test_idx)
    X_train = {}
    X_test = {}

    for view in X.keys():
        X_train[view] = X[view][train_idxs[idx_split]]
        X_test[view] = X[view][test_idxs[idx_split]]

    y_train = y[train_idxs[idx_split]]
    y_test = y[test_idxs[idx_split]]

    return X_train, y_train, X_test, y_test
















