import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):

    def __init__(self, in_features, out_features, hidden_features=[], batchNorm=False,
                 nonlinearity='leaky_relu', negative_slope=0.01,
                 with_output_nonlinearity=True):
        """
        初始化多层感知机（MLP）模型的类。

        参数:
        - in_features (int): 输入特征的维度。
        - out_features (int): 输出特征的维度。
        - hidden_features (list, optional): 隐藏层的特征维度列表，默认为空列表。
        - batchNorm (bool, optional): 是否使用批归一化，默认为 False。
        - nonlinearity (str, optional): 非线性激活函数类型，默认为 'leaky_relu'。
        - negative_slope (float, optional): Leaky ReLU 激活函数的负斜率，默认为 0.1。
        - with_output_nonlinearity (bool, optional): 是否在输出层应用激活函数，默认为 True。

        属性:
        - self.nonlinearity (str): 非线性激活函数类型。
        - self.negative_slope (float): Leaky ReLU 激活函数的负斜率。
        - self.fcs (ModuleList): 存储线性层和激活函数的列表。

        逻辑:
        - 如果存在隐藏层 (hidden_features 不为空)，则创建多个线性层和激活函数，并添加到 self.fcs 中。
          - 输入层到第一个隐藏层使用 in_features 到 hidden_features[0] 的线性层。
          - 隐藏层之间使用 hidden_features[i] 到 hidden_features[i+1] 的线性层。
          - 最后一个隐藏层到输出层使用 hidden_features[-1] 到 out_features 的线性层。
          - 对每个线性层，如果 with_output_nonlinearity 为 True 或者是最后一个隐藏层之前的层，
            则添加相应的激活函数（ReLU 或 Leaky ReLU）。
          - 如果启用批归一化 (batchNorm 为 True)，则在每个线性层后添加 Batch Normalization。
        - 如果没有隐藏层，则只创建一个输入层到输出层的线性层，并按照相同的逻辑添加激活函数和批归一化。

        最后，调用 self.reset_parameters() 来初始化参数。
        """
        super(MLP, self).__init__()
        self.nonlinearity = nonlinearity
        self.negative_slope = negative_slope
        self.fcs = nn.ModuleList()


        # 如果存在隐藏层，设置输入和输出维度
        if hidden_features:
            in_dims = [in_features] + hidden_features
            out_dims = hidden_features + [out_features]

            # 遍历隐藏层，创建线性层和激活函数
            for i in range(len(in_dims)):

                self.fcs.append(nn.Linear(in_dims[i], out_dims[i]))

                # 如果是输出层或者是隐藏层之前的层，添加激活函数
                if with_output_nonlinearity or i < len(hidden_features):
                    # 如果启用批归一化，添加 Batch Normalization
                    if batchNorm:
                        self.fcs.append(nn.BatchNorm1d(out_dims[i], track_running_stats=True))

                    # 根据激活函数类型添加相应的激活函数
                    if nonlinearity == 'relu':
                        self.fcs.append(nn.ReLU(inplace=True))
                    elif nonlinearity == 'leaky_relu':
                        self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                    else:
                        raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))
        else:
            # 如果没有隐藏层，只创建一个输入层到输出层的线性层，并按照相同的逻辑添加激活函数和批归一化
            self.fcs.append(nn.Linear(in_features, out_features))
            # self.fcs.append(nn.Dropout(0.5))
            if with_output_nonlinearity:
                if batchNorm:
                    self.fcs.append(nn.BatchNorm1d(out_features, track_running_stats=True))
                if nonlinearity == 'relu':
                    self.fcs.append(nn.ReLU(inplace=True))
                elif nonlinearity == 'leaky_relu':
                    self.fcs.append(nn.LeakyReLU(negative_slope, inplace=True))
                else:
                    raise ValueError("Unsupported nonlinearity {}".format(nonlinearity))

        # 调用 reset_parameters() 来初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        重置模型参数的方法。

        逻辑:
        - 对于模型中的每个层 (l) 进行遍历。
          - 如果该层是线性层 (nn.Linear)，则使用 Kaiming 初始化方法初始化权重。
            - 对于 Leaky ReLU 或 ReLU 激活函数，初始化偏置项为均匀分布在 [0, 0.1] 范围内。
            - 对于其他激活函数，初始化偏置项为常数 0.0。
          - 如果该层是 Batch Normalization 层 (nn.BatchNorm1d)，则调用其 reset_parameters 方法。

        注意:
        - Kaiming 初始化适用于使用 ReLU 或 Leaky ReLU 激活函数的网络。
        - Batch Normalization 层会自动进行参数初始化，因此不需要额外处理。
        """
        for l in self.fcs:
            # 如果是线性层
            if l.__class__.__name__ == 'Linear':
                # 使用 Kaiming 初始化方法初始化权重
                nn.init.kaiming_uniform_(l.weight, a=self.negative_slope, nonlinearity=self.nonlinearity)

                # 初始化偏置项
                if self.nonlinearity == 'leaky_relu' or self.nonlinearity == 'relu':
                    nn.init.uniform_(l.bias, 0, 0.1)
                else:
                    nn.init.constant_(l.bias, 0.0)
            # 如果是 Batch Normalization 层
            elif l.__class__.__name__ == 'BatchNorm1d':
                # 调用 Batch Normalization 层的 reset_parameters 方法
                l.reset_parameters()

    def forward(self, input):

        """
        定义模型的前向传播方法。

        参数:
        - input: 输入数据。

        返回:
        - output: 经过模型前向传播后的输出。

        逻辑:
        - 对于模型中的每个层 (l)，将输入数据传递给该层，更新输入数据。
        - 返回最后一层的输出作为模型的最终输出。

        注意:
        - 此方法简单地将输入数据通过模型的每一层传递，并返回最终结果。
        """
        for l in self.fcs:
            input = l(input)
        return input


class GINLayer(nn.Module):
    def __init__(self, mlp, eps=0.0, train_eps=True, residual=True):
        """
        GINLayer 类的初始化方法。

        参数:
        - mlp (MLP): 用于处理节点特征的多层感知机。
        - eps (float, optional): 图神经网络（GIN）中的 epsilon 参数，默认为 0.0。
        - train_eps (bool, optional): 是否训练 epsilon 参数，默认为 True。
        - residual (bool, optional): 是否使用残差连接，默认为 True。

        属性:
        - self.mlp (MLP): 用于处理节点特征的多层感知机。
        - self.initial_eps (float): epsilon 参数的初始值。
        - self.residual (bool): 是否使用残差连接。
        - self.eps (torch.nn.Parameter or torch.Tensor): epsilon 参数，可以是可训练的 Parameter 或者固定的 Tensor。

        逻辑:
        - 初始化时设置 mlp、eps、train_eps、residual 等属性，并调用 reset_parameters 方法进行参数初始化。
        """
        super(GINLayer, self).__init__()
        self.mlp = mlp
        self.initial_eps = eps
        self.residual = residual

        # 如果训练 epsilon，则创建可训练的 Parameter；否则，创建固定的 Tensor。
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer('eps', torch.Tensor([eps]))

        # 调用 reset_parameters() 来初始化参数
        self.reset_parameters()

    def reset_parameters(self):
        """
        重置模型参数的方法。

        逻辑:
        - 调用 mlp 的 reset_parameters 方法初始化多层感知机的参数。
        - 设置 epsilon 参数的值为初始值。
        """
        self.mlp.reset_parameters()
        self.eps.data.fill_(self.initial_eps)

    def forward(self, input, adj):
        """
        定义模型的前向传播方法。
        参数:
        - input: 输入节点特征。
        - adj: 图的邻接矩阵。
        返回:
        - output: 经过模型前向传播后的输出。
        逻辑:
        - 将输入节点特征保存为 res。
        - 使用邻接矩阵对邻居节点信息进行聚合。
        - 对中心节点表示进行重新加权。
        - 使用 mlp 处理节点表示。
        - 如果启用残差连接，则将输出与输入相加，否则直接返回输出。
        """
        res = input


        adj = adj.float()
        res = res.float()
        neighs = torch.matmul(adj, res)

        res = (1 + self.eps) * res + neighs

        res = self.mlp(res)

        if self.residual:
            output = res + input
        else:
            output = res

        return output


class GIN(nn.Module):
    # def __init__(self, num_layers, in_features, out_features, hidden_features=[],
    #              eps=0.0, train_eps=True, residual=True, batchNorm=True,
    #              nonlinearity='leaky_relu', negative_slope=0.01):
    def __init__(self, num_layers, in_features, out_features, hidden_features=[],
                 eps=0.0, train_eps=True, residual=True, batchNorm=True,
                 nonlinearity='leaky_relu', negative_slope=0.01):

        """
        GIN 类的初始化方法。

        参数:
        - num_layers (int): GIN 的层数。
        - in_features (int): 输入特征的维度。
        - out_features (int): 输出特征的维度。
        - hidden_features (list, optional): 隐藏层的特征维度列表，默认为空列表。
        - eps (float, optional): GINLayer 中的 epsilon 参数，默认为 0.0。
        - train_eps (bool, optional): 是否训练 epsilon 参数，默认为 True。
        - residual (bool, optional): 是否使用残差连接，默认为 True。
        - batchNorm (bool, optional): GINLayer 中是否使用批归一化，默认为 True。
        - nonlinearity (str, optional): GINLayer 中的非线性激活函数类型，默认为 'leaky_relu'。
        - negative_slope (float, optional): GINLayer 中 Leaky ReLU 激活函数的负斜率，默认为 0.1。

        属性:
        - self.GINLayers (ModuleList): 存储 GINLayer 层的列表。

        逻辑:
        - 根据输入和输出特征的维度，确定第一层是否使用残差连接。
        - 创建 GINLayer 层，并添加到 self.GINLayers 中。
        - 根据 num_layers 创建额外的 GINLayer 层，添加到 self.GINLayers 中。
        - 调用 reset_parameters 方法来初始化参数。
        """
        super(GIN, self).__init__()

        self.GINLayers = nn.ModuleList()

        # 确定第一层是否使用残差连接
        if in_features != out_features:
            first_layer_res = False
        else:
            first_layer_res = True


        self.GINLayers.append(GINLayer(MLP(in_features, out_features, hidden_features, batchNorm,
                                           nonlinearity, negative_slope),
                                       eps, train_eps, first_layer_res))


        for i in range(num_layers - 1):
            self.GINLayers.append(GINLayer(MLP(out_features, out_features, hidden_features, batchNorm,
                                               nonlinearity, negative_slope),
                                           eps, train_eps, residual))

        self.reset_parameters()

    def reset_parameters(self):

        for l in self.GINLayers:
            l.reset_parameters()

    def forward(self, input, adj):

        for l in self.GINLayers:
            input = l(input, adj)

        return input

class LearnModel(nn.Module):
    def __init__(self, input_dims):
        super(LearnModel, self).__init__()
        self.input_sizes = input_dims
        self.weidu = 128
        self.device = torch.device('cuda')
        self.fc = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, self.weidu)
            )
            for dim in input_dims
        ])

    def forward(self, x):
        for key in x:
            x[key] = x[key].to(self.device)
        outs = [fc_demo(view) for fc_demo, view in zip(self.fc, x.values())]

        return outs


class FDModel(nn.Module):

    def __init__(self, label_nums, view_classes, input_size, in_features_y, hidden_features):

        super(FDModel, self).__init__()
        self.input_sizes = input_size
        self.weidu = 128
        self.device = torch.device('cuda')
        self.view_nums = view_classes
        self.label_nums = label_nums
        self.class_emb = in_features_y


        self.bn1 = nn.Sequential(
            nn.BatchNorm1d(self.label_nums),
            nn.ReLU(),
        )

        self.NN3 = nn.Sequential(
            nn.Linear(self.class_emb, hidden_features),
            nn.Linear(hidden_features, self.weidu),
            nn.Sigmoid()
        )

        self.multi_cls = nn.ModuleList([MultiViewClass(self.label_nums, self.weidu) for _ in range(self.view_nums)])

    def forward(self, x, y, view_adj):

        y = self.NN3(y) # b2 x h
        a = y.unsqueeze(0)

        z_tmp = []
        for i, x_demo in enumerate(x):
            z_tmp.append(x_demo.unsqueeze(1) * (1 - a[0][i]))

        z = []
        for i in range(self.view_nums):
            z.append(self.multi_cls[i](z_tmp[i].squeeze(1)))

        w_tmp = []
        for i in range(self.view_nums):
            w_tmp.append(self.DM(z[i], self.label_nums))

        rc = []

        for i in range(len(w_tmp)):
            others = [w_tmp[j] for j in range(len(w_tmp)) if j != i]
            rc_i = torch.min(w_tmp[i] * (self.view_nums - 1) / sum(others), torch.tensor(1.0))
            rc.append(rc_i)

        output = 0
        for i in range(len(rc)):
            output += rc[i] * z[i]

        fu = output
        fu = self.bn1(fu.squeeze(1))

        return fu

    def DM(self, fm, label_num):
        """
        计算公式 DUm = 1/C * sum(|Softmax(fm(xm))_i - μ|) 使用 PyTorch 实现

        :param logits: 模型输出的 logits，形状为 (C,)
        :param mu: 均值 μ，标量
        :return: 计算得到的损失 DUm
        """
        # 计算 Softmax
        softmax_outputs = F.softmax(fm, dim=1)
        mu = 1 / label_num

        # 计算绝对误差
        abs_diff = torch.abs(softmax_outputs - mu)
        # 计算平均值
        loss = torch.mean(abs_diff)
        return loss

class MultiViewClass(nn.Module):
    def __init__(self, label_nums, input_dim=128):
        super(MultiViewClass, self).__init__()
        self.cls = nn.Sequential(
            nn.BatchNorm1d(input_dim),
            nn.ReLU(),
            nn.Linear(input_dim, label_nums)
        )

    def forward(self, x):
        out = self.cls(x)
        return out


