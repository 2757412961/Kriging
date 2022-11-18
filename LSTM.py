'''
    Name: Long Short Term Memory
    Creation: 2020-03-04
'''

from NNUtil import *

import numpy as np
import matplotlib.pyplot as plt
import gc

# Const Variable
from ConstVariable import *


# 神经网络模型
class LSTM_Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layer, output_size):
        super(LSTM_Model, self).__init__()
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, hidden_size * 2)
        self.layer3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.layer4 = nn.Linear(hidden_size * 2, hidden_size)
        self.layer5 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):  # torch.Size([seq, batch_size, hidden_size])
        x, _ = self.layer1(x)
        s, b, h = x.size()  # torch.Size([826, 1, 12])
        x = x.view(s * b, h)  # torch.Size([826, 12])
        x = self.layer2(F.selu(self.dropout(x)))  # torch.Size([826, 12])
        x = self.layer3(F.relu(self.dropout(x)))  # torch.Size([826, 12])
        x = self.layer4(F.selu(self.dropout(x)))  # torch.Size([826, 12])
        x = self.layer5(F.selu(self.dropout(x)))  # torch.Size([826, 1])
        x = x.view(s, b, -1)  # torch.Size([826, 1, 1])

        return x


# trans str to double
def transDataStr2Float(data):
    return [[float(col) for col in row] for row in data]


# sort seq by index
def sortByIndex(data, index=3):
    data = sorted(data, key=lambda x: x[index])
    data = np.array(data)
    tempAllT = data[:, index]
    tempAllT = tempAllT - firstDay + 1
    data[:, index] = tempAllT
    # print(data)

    data = np.array(data)
    return data


# 获得训练集合测试集
def getTrainAndTest(data, input_size=4):
    data = transDataStr2Float(data)

    # 按时间排序
    data = sortByIndex(data, 3)

    # 设置输入输出数据集
    data_X = data[:, :input_size]
    data_Y = data[:, input_size]

    # 设置训练集和测试集
    train_X, test_X = divideDataset(data_X, 10)
    train_Y, test_Y = divideDataset(data_Y, 10)

    # 设置LSTM模型数据类型形状
    train_X = torch.from_numpy(train_X).view(-1, 1, input_size)
    test_X = torch.from_numpy(test_X).view(-1, 1, input_size)
    train_Y = torch.from_numpy(train_Y).view(-1, 1, 1)
    test_Y = torch.from_numpy(test_Y).view(-1, 1, 1)

    return train_X.float(), test_X.float(), train_Y.float(), test_Y.float()


# get data boundary
def getIntBox(data):
    x0 = np.array(data)[:, 0]
    y0 = np.array(data)[:, 1]
    z0 = np.array(data)[:, 2]
    t0 = np.array(data)[:, 3]

    return int(min(x0)), int(max(x0)) + 1, int(min(y0)), int(max(y0)) + 1


def run(data, columns):
    input_size = len(columns) - 1
    hidden_size = 12
    num_layer = 3
    output_size = 1
    lstm = LSTM_Model(input_size, hidden_size, num_layer, output_size)

    train_X, test_X, train_Y, test_Y = getTrainAndTest(data, input_size)
    trained_lstm, accuracy = train(lstm, train_X, test_X, train_Y, test_Y, 500)

    del data, columns, lstm
    gc.collect()

    return trained_lstm, accuracy


def Grid(Z, T):
    import Utils
    deltaZ, deltaT, intervel = 25, 120, 2
    minX, maxX, minY, maxY = getIntBox(Utils.getTemperatureByZT(Z, deltaZ, T, deltaT)[0])
    print("Boundary: ", minX, maxX, minY, maxY)

    outputData, outputColumns = [], ['x', 'y', 'v']
    for x in np.arange(minX, maxX, 2):
        for y in np.arange(minY, maxY, 2):
            data, columns = Utils.getTemperatureByXYZT(x, 5, y, 5, Z, deltaZ, T, deltaT)
            if len(data) < 75:
                continue

            trained_lstm, accuracy = run(data, columns)
            if accuracy < 0.75:
                continue

            v = trained_lstm(torch.tensor([x, y, Z, T - firstDay + 1]).view(-1, 1, 4).float())
            v = v.view(-1).detach().numpy()[0]
            outputData.append([x, y, v])

            del data, columns, trained_lstm, v
            gc.collect()

    return outputData, outputColumns
