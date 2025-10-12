import sys
import argparse

sys.path.append(".\Cores")

from Model import AssoDMVC
from dataset import *
from cross_validation import *
from utils import Init_random_seed, clearOldLogs
from default_configs import *
import os.path

import torch


parser = argparse.ArgumentParser()
parser.add_argument('exp',
                    help='name of experiment')
parser.add_argument('--dataset', '-dataset', type=str, default="nus_wide",
                    help='dataset on which experiment is conducted')
parser.add_argument('--batch_size', '-bs', type=int, default=128,
                    help='batch size for one iteration during training')
parser.add_argument('--lr', '-lr', type=float, default=1e-3,
                    help='learning rate parameter')
parser.add_argument('--weight_decay', '-wd', type=float, default=1e-5,
                    help='weight decay parameter')
parser.add_argument('--max_epoch', '-max_epoch', type=int, default=200,
                    help='maximal training epochs')
parser.add_argument('--lambda_tradeoff', '-lambda', type=float, default=1.0,
                    help='trade-off parameter for embedding loss')
parser.add_argument('--class_emb', '-class_emb', type=int, default=256,
                    help='dimensionality of label embedding')
parser.add_argument('--in_layers', '-in_layers', type=int, default=1,
                    help='number of layers for obtaining latent representation')
parser.add_argument('--cuda', '-cuda', action='store_true',
                    help='whether to use gpu')
parser.add_argument('--quiet', '-quiet', action='store_true',
                    help='whether to train in quiet mode')
parser.add_argument('--ada_epoch', '-ada_epoch', action='store_true',
                    help='whether to decide the max_epoch on the first fold during cross-validation')
parser.add_argument('--default_cfg', '-default_cfg', action='store_true',
                    help='whether to run experiment with default hyperparameters')

if __name__ == '__main__':
    args = parser.parse_args()

    # Setting random seeds
    Init_random_seed()

    # Loading dataset
    dataset_name = args.dataset
    dataset = eval(dataset_name)()
    input_size = []
    X_train, y_train, X_test, y_test = dataset.data()
    for key in X_train.keys():
        input_size.append(X_train[key].size(1))

    # Setting configurations
    configs = generate_default_config()

    configs['dataset'] = dataset

    configs['view_classes'] = 7
    configs['label_nums'] = 10


    configs['input_size'] = input_size

    # 视图间关系嵌入的维度
    configs['class_emb'] = args.class_emb

    configs['in_layers'] = args.in_layers

    configs['lambda'] = args.lambda_tradeoff
    configs['lr'] = args.lr
    configs['weight_decay'] = args.weight_decay
    configs['batch_size'] = args.batch_size
    configs['max_epoch'] = args.max_epoch
    configs['ada_epoch'] = args.ada_epoch
    configs['use_gpu'] = args.cuda
    configs['device'] = torch.device('cuda' if torch.cuda.is_available() and configs['use_gpu'] else 'cpu')
    configs['exp'] = args.exp
    configs['model_name'] = 'CLIFModel'

    # Clear old logs
    clearOldLogs(os.path.join(configs['model_name'], configs['exp']))

    # Creating model
    model = AssoDMVC(configs).to(configs['device'])


    # Cross-validation
    cross_validation(model, dataset, configs, quiet_mode=args.quiet)





