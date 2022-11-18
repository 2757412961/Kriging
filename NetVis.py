# from NNUtil import *
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchviz import make_dot

from tensorboardX import SummaryWriter

import sys
import torch
import tensorwatch as tw
import torchvision.models

class Net_Model(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net_Model, self).__init__()
        self.inLayer = torch.nn.Linear(n_input, n_hidden)
        self.hidLayer1 = torch.nn.Linear(n_hidden, n_hidden*2)
        self.hidLayer2 = torch.nn.Linear(n_hidden*2, n_hidden*2)
        self.hidLayer3 = torch.nn.Linear(n_hidden*2, n_hidden)
        self.outLayer = torch.nn.Linear(n_hidden, n_output)
        self.dropout = nn.Dropout(0.5)
        self.net = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.Linear(n_hidden, n_output)
        )


    def forward(self, X):  # torch.Size([90, 1, 1])
        X = self.inLayer(F.selu(self.dropout(X)))  # torch.Size([90, 1, 12])
#         X = self.hidLayer1(F.selu(self.dropout(X)))  # torch.Size([90, 1, 12])
#         X = self.hidLayer2(F.selu(self.dropout(X)))  # torch.Size([90, 1, 12])
#         X = self.hidLayer3(F.selu(self.dropout(X)))  # torch.Size([90, 1, 12])
        X = self.outLayer(F.selu(self.dropout(X)))  # torch.Size([90, 1, 1])

        # X = self.net(X)
        return X


# LSTM模型
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, output_size):
        super(LSTM_Model, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer, bias=True)
        self.layer2 = nn.Linear(int(hidden_size*1), int(hidden_size*4))
        self.layer3 = nn.Linear(int(hidden_size*4), int(hidden_size*8))
        self.layer4 = nn.Linear(int(hidden_size*8), int(hidden_size*4))
        self.layer5 = nn.Linear(int(hidden_size*4), int(hidden_size*2))
        self.layer6 = nn.Linear(int(hidden_size*2), int(hidden_size*1))
        self.layer7 = nn.Linear(int(hidden_size*1), output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):  # torch.Size([seq, batch_size, hidden_size])
        x, _ = self.layer1(x)
        s, b, h = x.size()  # torch.Size([826, 1, 12]) [seq_len, batch, hidden_size]
        x = x.view(s * b, h)  # torch.Size([826, 12])
        x = self.layer2(x)  # torch.Size([826, 12])
        x = self.layer3(x)  # torch.Size([826, 12])
        x = self.layer4(x)  # torch.Size([826, 12])
        x = self.layer5(x)  # torch.Size([826, 12])
        x = self.layer6(x)  # torch.Size([826, 12])
        x = self.layer7(x)  # torch.Size([826, 1])
        x = x.view(s, b, -1)  # torch.Size([826, 1, 1])

        return x

# model = Net_Model(12, 32, 1)
model = LSTM_Model(12, 64, 2, 1)
x = torch.randn(20, 1, 12)

# 方法一 本地
# vis_graph = make_dot(model(x), params=dict(model.named_parameters()))
# vis_graph.view()

# 方法二 pytorchvis中
# with SummaryWriter(comment='Net_Model') as w:
#     w.add_graph(model, (x, ))

# 方法三 Jupyter notebook
# 在notebook中!!!
# 传入三个参数，
# 第一个为model，
# 第二个参数为input_shape，
# 第三个参数为orientation，可以选择’LR’或者’TB’，分别代表左右布局与上下布局。
# https://lutzroeder.github.io/netron/
# tw.draw_model(model, [20, 1, 12])



exit(0)