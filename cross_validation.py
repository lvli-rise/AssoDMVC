import os.path
import torch
from sklearn.model_selection import StratifiedKFold, KFold
from torch.autograd import Variable
from Model import AssoDMVC

from dataset import dataset2

def cross_validation3(model, dataset, configs, quiet_mode=True):
    X_train, y_train, X_test, y_test = dataset.data()
    model.Train_1(X_train, y_train, X_test, y_test, [], quiet_mode=quiet_mode)










        


