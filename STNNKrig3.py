'''
    Name: Spatio-temporal Kriging
    Creation: 2020-03-06
'''

from NNUtil import *

import numpy as np
import matplotlib.pyplot as plt
import gc

# Const Variable
from ConstVariable import *


####################################################################################
# NN模型
class Net_Model(torch.nn.Module):
    def __init__(self, n_input, n_hidden, n_output):
        super(Net_Model, self).__init__()
        self.inLayer = torch.nn.Linear(n_input, n_hidden)
        self.hidLayer = torch.nn.Linear(n_hidden, n_hidden)
        self.outLayer = torch.nn.Linear(n_hidden, n_output)
        self.dropout = nn.Dropout(0.5)

    def forward(self, X):  # torch.Size([90, 1, 1])
        X = self.inLayer(F.selu(self.dropout(X)))  # torch.Size([90, 1, 12])
        X = self.hidLayer(F.selu(self.dropout(X)))  # torch.Size([90, 1, 12])
        X = self.hidLayer(F.selu(self.dropout(X)))  # torch.Size([90, 1, 12])
        X = self.outLayer(F.selu(self.dropout(X)))  # torch.Size([90, 1, 1])

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
        x = self.layer2(F.selu(self.dropout(x)))  # torch.Size([826, 12])
        x = self.layer3(F.relu(self.dropout(x)))  # torch.Size([826, 12])
        x = self.layer4(F.selu(self.dropout(x)))  # torch.Size([826, 12])
        x = self.layer5(F.selu(self.dropout(x)))  # torch.Size([826, 12])
        x = self.layer6(F.selu(self.dropout(x)))  # torch.Size([826, 12])
        x = self.layer7(F.selu(self.dropout(x)))  # torch.Size([826, 1])
        x = x.view(s, b, -1)  # torch.Size([826, 1, 1])

        return x


# CNN Kriging
class STCNNKrig(nn.Module):
    def __init__(self, output_size):
        super(STCNNKrig, self).__init__()
        self.output_size = output_size
        self.layer1 = nn.Conv2d(in_channels=1,
                                out_channels=3,
                                kernel_size=3,
                                stride=1,
                                padding=1,
                                groups=1)
        self.layer2 = nn.MaxPool2d(kernel_size=2, padding=1)
        self.layer3 = nn.Linear(3 * (int(output_size/2)+1) * 3, 2048)
        self.layer4 = nn.Linear(2048, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):  # torch.Size([batch, channel, Height, Width])
        # print(x.size()) # torch.Size([1, 1, n, 5])
        x = self.layer1(x) # torch.Size([1, 3, n, 5])
        x = self.layer2(F.selu(self.dropout(x))) # torch.Size([1, 3, int(n/2)+1, 3])
        b, c, h, w = x.size()
        x = x.view(b, c * h * w) # torch.Size([1, c * h * w])
        x = self.layer3(F.selu(self.dropout(x))) # torch.Size([1, 2048])
        x = self.layer4(F.relu(self.dropout(x))) # torch.Size([1, n])
        x = x.view(b, self.output_size)

        return x


####################################################################################
# calculate Distance
def calSDis(x1, y1, x2, y2):
    return np.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2) + 1


def calTDis(t1, t2):
    return np.abs(t1 - t2) + 1


def calSemi(v1, v2):
    return np.power((v1 - v2), 2) / 2


# Divide the list into n blocks
def divideByValue(x0, y0, n):
    minX, maxX, diff = min(x0), max(x0), (max(x0) - min(x0)) / n
    tempX, tempY = [[] for i in range(n)], [[] for i in range(n)]
    for i in range(len(x0)):
        for j in range(n):
            if minX + diff * j <= x0[i] < minX + diff * (j + 1):
                tempX[j].append(x0[i])
                tempY[j].append(y0[i])
                break

    emptyCount = 0
    for i in range(n):
        if len(tempY[i - emptyCount]) == 0:
            del tempX[i - emptyCount], tempY[i - emptyCount]
            emptyCount += 1

    tempX = [np.mean(x) for x in tempX]
    tempY = [np.mean(y) for y in tempY]

    return tempX, tempY


# sort seq by index
def sortByIndex(data, index=3):
    data = sorted(data, key=lambda x: x[index])
    data = np.array(data)

    return data


def Manhattan(lst, x, y, points):
    for point in points:
        lst.append(x - point[0])
        lst.append(y - point[1])
        lst.append(calSDis(x, y, point[0], point[1]))

    return lst


# 时空变异函数
def createYst(data):
    seq_len = len(data)
    batch = 1
    input_size = 3 + 3 * 4
    hidden_size = 64
    num_layer = 1
    output_size = 1
    model = LSTM_Model(input_size, hidden_size, num_layer, output_size)
    print(model)

    # 按照时间排序
    data = sortByIndex(data)

    # 按照值平均分割数据为200份
    lon = 145
    lat = 0
    depth = 250
    day = 25440 - firstDay + 1
    points = [(lon, lat), (130, 20), (165, -30), (160, 40)]
    X, Y = [], []
    for i in range(seq_len):
        x = data[i][0]
        y = data[i][1]
        # z = data[i][2]
        # dz = z - depth
        t = data[i][3]
        dt = t - day

        lst = [x, y, dt]
        lst = Manhattan(lst, x, y, points)
        X.append(lst)
        Y.append([data[i][4]])

    # 按照值平均分割数据为200份
    train_X, test_X = divideDataset(X, 10)
    train_Y, test_Y = divideDataset(Y, 10)

    # 设置LSTM模型数据类型形状
    train_X = torch.from_numpy(train_X).view(-1, batch, input_size).float()
    test_X  = torch.from_numpy(test_X) .view(-1, batch, input_size).float()
    train_Y = torch.from_numpy(train_Y).view(-1, batch, output_size).float()
    test_Y  = torch.from_numpy(test_Y) .view(-1, batch, output_size).float()

    # 训练模型
    model, accuracy = train(model, train_X, test_X, train_Y, test_Y, 600)

    # 定义函数
    def Yst(x, y, z, t):
        dt = t - day

        lst = [x, y, dt]
        lst = Manhattan(lst, x, y, points)

        X_input = np.array(lst)
        input = torch.from_numpy(X_input).view(-1, batch, input_size).float()
        out = model(input)
        return out.reshape(-1).detach().numpy()[0]

    return Yst


def drawResultYt(model, X, Y):
    plt.title('Result Yt Curve')

    x0 = X[:, 0, 2].reshape(-1).detach().numpy()
    y0 = Y.reshape(-1).detach().numpy()
    plt.scatter(x0, y0, label='OriginPoint')

    x_plt = x0
    y_plt = model(X)
    y_plt = y_plt.reshape(-1).detach().numpy()
    plt.plot(x_plt, y_plt, color='#FFA500', label='svCurve')

    plt.legend()
    plt.show()


####################################################################################
# trans str to double
def transDataStr2Float(data):
    fdata = np.array([[float(col) for col in row] for row in data])
    lstT = fdata[:, 3] - firstDay + 1
    fdata[:, 3] = lstT

    return fdata


def SpatioTemporalNeuralNetworkKriging(data, columns, depth, day):
    outputData, outputColumns = [], ['x', 'y', 'v']

    # 数据清洗，转换时间
    data = transDataStr2Float(data)
    print("Data  Length: ", len(data))

    # 建立模型
    Yst = createYst(data)  # 变异函数 计算Yst

    # 建立网格场，用神经网络和克里金估算每个点的值
    minX, maxX, minY, maxY = getBoundary()
    # v0 = np.array(data)[:, 4]
    print("Boundary: ", minX, maxX, minY, maxY)
    for m in np.arange(minX, maxX, 2):
        for n in np.arange(minY, maxY, 2):
            v = Yst(m, n, depth, day)

            outputData.append([m, n, v])
            print(m, n, v)

    return outputData, outputColumns
