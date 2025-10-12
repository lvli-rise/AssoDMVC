import torch
from torchmetrics import Precision, Recall, F1Score
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset


class ModelEngine():
    '''
    Engine to optimize CLIFModel.
    '''

    def __init__(self, configs={}):
        self.configs = configs
        # Checking configurations
        if self.configs['use_gpu'] and not torch.cuda.is_available():
            self.configs['use_gpu'] = False
        self.configs['use_multi_gpu'] = self.configs['use_gpu'] and \
                                        torch.cuda.device_count() > 1
        self.configs['device'] = torch.device('cuda' if torch.cuda.is_available()
                                                        and self.configs['use_gpu'] else 'cpu')

        self.batchsize = configs['batch_size']
        self.best_acc = 0
        self.best_precision = 0
        self.best_recall = 0
        self.best_F1 = 0

        self.state = {}

    def Learn(self, model, criterion, X_train, y_train, X_test, y_test, quiet_mode=False):

        model.to(self.configs['device'])
        y_train = y_train.to(self.configs['device'])
        y_test = y_test.to(self.configs['device'])

        self.state['adj'] = model.view_adj.data

        # Defining optimizer
        optimizer = torch.optim.Adam(model.get_config_optim(), lr=self.configs['lr'],
                                     weight_decay=self.configs['weight_decay'])
        epoch = self.configs['max_epoch']
        for i in range(epoch):
            print(f'Epoch [{i + 1}/{epoch}]')
            self.state['lr'] = self.configs['lr']

            # Training for one epoch
            self.train(model, criterion, X_train, y_train, i, optimizer)

            # 测试
            with torch.no_grad():
                output_test, _ = model(X_test)

                cls_loss_test = criterion[0](output_test.squeeze(), y_test.long())

                _, predicted_labels = torch.max(output_test.data, 1)
                correct_predictions = (predicted_labels.squeeze() == y_test).sum().item()

                total_samples = len(y_test)
                accuracy = correct_predictions / total_samples
                precision_metric = Precision(task="multiclass", average='macro', num_classes=self.configs['label_nums']).to(self.configs['device'])
                recall_metric = Recall(task="multiclass", average='macro', num_classes=self.configs['label_nums']).to(self.configs['device'])
                F1 = F1Score(task="multiclass", num_classes=self.configs['label_nums']).to(self.configs['device'])
                precision = precision_metric(predicted_labels, y_test)
                recall = recall_metric(predicted_labels, y_test)
                f1 = F1(predicted_labels, y_test)
                if accuracy > self.best_acc:
                    self.best_acc = accuracy
                    self.best_precision = precision
                    self.best_recall = recall
                    self.best_F1 = f1

        print("acc:{}".format(self.best_acc))
        print("precision:{}".format(self.best_precision))
        print("recall:{}".format(self.best_recall))
        print("F1:{}".format(self.best_F1))

    def train(self, model, criterion, X_train, y_train, i, optimizer):

        train_datasets = {}

        for key, value in X_train.items():
            train_datasets[key] = value
        train_dataset = TensorDataset(*train_datasets.values(), y_train)
        batch_size = self.batchsize
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        X_train_batch = {}
        for batch_idx, (data1, data2, data3, data4, data5, data6, data7, target) in enumerate(train_loader):
            X_train_batch['view1'] = data1
            X_train_batch['view2'] = data2
            X_train_batch['view3'] = data3
            X_train_batch['view4'] = data4
            X_train_batch['view5'] = data5
            X_train_batch['view6'] = data6
            X_train_batch['view7'] = data7

            self.Step(True, model, criterion, X_train_batch, target, i, batch_idx, optimizer)

    def Step(self, training, model, criterion, X, y, i, batch_idx, optimizer=None):
        '''
        Forwarding model and optimizing model once during training.
        '''
        optimizer.zero_grad()
        output, emb = model(X)

        cls_loss = criterion[0](output.squeeze(), y.long())
        embedding_loss = criterion[1](emb, self.state['adj'])
        loss = cls_loss + self.configs['lambda'] * embedding_loss

        _, predicted_labels = torch.max(output.data, 1)

        correct_predictions = (predicted_labels.squeeze() == y).sum().item()

        total_samples = len(y)

        accuracy = correct_predictions / total_samples
        loss.backward()
        optimizer.step()


