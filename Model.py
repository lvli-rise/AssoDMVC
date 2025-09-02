import torch
import torch.nn as nn
import math
import random
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 使用非图形界面后端，适合服务器

from layers import *
from loss import *
from engine import ModelEngine
from utils import Init_random_seed

class Net(nn.Module):
    def __init__(self, configs):
        super(Net, self).__init__()
        self.rand_seed = configs['rand_seed']
        self.label_nums = configs['label_nums']
        self.view_nums = configs['view_classes']


        # Label semantic encoding module
        self.view_embedding = nn.Parameter(torch.eye(self.view_nums),
                                            requires_grad=False)

        self.view_adj = nn.Parameter(torch.eye(self.view_nums),
                                      requires_grad=False)
        self.GIN_encoder = GIN(2, self.view_nums, configs['class_emb'], [math.ceil(configs['class_emb'] / 2)])
        self.Learn_model = LearnModel(configs['input_size'])
        self.FD_model = FDModel(configs['label_nums'], configs['view_classes'], configs['input_size'], configs['class_emb'], configs['class_emb'], 128, configs['in_layers'], 1, False, 'leaky_relu', 0.1)
        self.reset_parameters()


    def reset_parameters(self):
        Init_random_seed(self.rand_seed)
        nn.init.normal_(self.view_embedding)

    def get_config_optim(self):
        return [
                {'params': self.GIN_encoder.parameters()},
                {'params': self.Learn_model.parameters()},
                {'params': self.FD_model.parameters()},
                ]


    def forward(self, input):

        # Generating semantic label embeddings via label semantic encoding module
        view_embedding = self.GIN_encoder(self.view_embedding, self.view_adj)
        input_FD_model = self.Learn_model(input)
        # Generating label-specific features via semantic-guided feature-disentangling module
        X = self.FD_model(input_FD_model, view_embedding, self.view_adj)
        return X, view_embedding



class AssoDMVC(nn.Module):
    def __init__(self, configs):
        super(AssoDMVC, self).__init__()
        self.configs = configs

        self.label_nums = configs['label_nums']
        self.view_nums = configs['view_classes']


        self.adj_label_num = self.label_nums * self.view_nums
        if self._configs('dtype') is None:
            self.configs['dtype'] = torch.float

        # Creating model
        self.model = Net(self.configs)
        self.device = torch.device('cuda')

        # Defining loss function
        self.criterion = nn.CrossEntropyLoss()
        self.emb_criterion = LinkPredictionLoss_cosine()

        # Creating learning engine
        self.engine = ModelEngine(self.configs)


    def _configs(self, name):
        if name in self.configs:
            return self.configs[name]


    def Train_1(self, X_train, y_train, X_test, y_test, train_index = [], quiet_mode=False):
        X_train, y_train = self.on_start_train(X_train, y_train, train_index)

        # Learning
        self.engine.Learn(self.model, [self.criterion, self.emb_criterion], X_train,
                          y_train, X_test, y_test, quiet_mode)


    def on_start_train(self, X_train, y_train, train_index):

        self.model.view_adj.data = self.sym_conditional_prob(X_train, y_train, train_index)
        return X_train, y_train

    def on_start_train_label(self, X_train, y_train):

        self.model.view_adj.data = self.sym_conditional_prob_label(X_train, y_train)
        return X_train, y_train

    def sym_conditional_prob(self, X_train, y_train, train_index):

        X_train_flod = {}

        if len(train_index) == 0:
            X_train_flod = X_train
        else:
            for name_of_view in X_train.keys():
                X_train_flod[name_of_view] = X_train[name_of_view][train_index]


        random.seed(42)
        length = int(len(X_train_flod['view1']) / 10)
        random_values = random.sample(range(len(X_train_flod['view1'])), length)

        euclidean_distances = []
        for name_of_view in X_train_flod.keys():
            temp = X_train_flod[name_of_view][random_values]
            distdistance = torch.cdist(X_train_flod[name_of_view], temp, p=2)
            euclidean_distances.append(distdistance)
        adj = []
        for matrix1 in euclidean_distances:
            for matrix2 in euclidean_distances:
                result = torch.sub(matrix1, matrix2)
                result_abs = torch.abs(result)
                mean_similarity = torch.mean(result_abs)
                adj.append(mean_similarity)
        adj = torch.tensor(adj, dtype=float)
        adj = torch.reshape(adj, (self.configs['view_classes'], self.configs['view_classes']))

        adj = self.symmetric_Normalization(adj)

        diag_idx = torch.arange(min(adj.size(0), adj.size(1)))
        adj[diag_idx, diag_idx] = 0
        return adj

    def dtype(self, dtype=None):

        '''
        Changing the dtype of parameters in the model
        '''
        if dtype is not None:
            self.configs['dtype'] = dtype
        self.model.type(self.configs['dtype'])



    def symmetric_Normalization(self, adj):
        
        row_sum = adj.sum(dim=1)
        d_inv_sqrt = torch.pow(row_sum, -0.5)
        d_inv_sqrt[d_inv_sqrt == float('inf')] = 0
        d_mat_inv_sqrt = torch.diag(d_inv_sqrt)

        
        adj = 1 - torch.mm(torch.mm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj






