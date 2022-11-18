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
        self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
        self.layer2 = nn.Linear(hidden_size, hidden_size)
        self.layer3 = nn.Linear(hidden_size, hidden_size)
        self.layer4 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):  # torch.Size([seq, batch_size, hidden_size])
        x, _ = self.layer1(x)
        s, b, h = x.size()  # torch.Size([826, 1, 12])
        x = x.view(s * b, h)  # torch.Size([826, 12])
        x = self.layer2(F.selu(self.dropout(x)))  # torch.Size([826, 12])
        x = self.layer3(F.relu(self.dropout(x)))  # torch.Size([826, 12])
        x = self.layer4(F.selu(self.dropout(x)))  # torch.Size([826, 1])
        x = x.view(s, b, -1)  # torch.Size([826, 1, 1])

        return x


# CNN Kriging
class STNNKrig(nn.Module):
    def __init__(self, output_size):
        super(STNNKrig, self).__init__()
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


    # 获得距离 对应 半方差值
def generateInput(data, x, y, z, t):
    input = []
    for i in range(len(data)):
        dx = np.abs(data[i][0] - x)
        dy = np.abs(data[i][1] - y)
        ds = calSDis(data[i][0], data[i][1], x, y)
        dp = calTDis(data[i][2], z)
        dt = calTDis(data[i][3], t)
        input.append([dx, dy, ds, dp, dt])

    return np.array(input)


# 时空变异函数
def createYst(data):
    n = len(data)
    in_channels = 1
    width = 5
    height = n
    output_size = n
    model = STNNKrig(output_size)

    # 按照时间排序
    data = sortByIndex(data)
    v0 = np.array(data)[:, 4]
    v0 = torch.from_numpy(v0).view(-1, 1).float()
    # 按照值平均分割数据为200份
    # train_X, test_X = divideDataset(X, 10)
    # train_Y, test_Y = divideDataset(Y, 10)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # 开始训练
    for e in range(10):
        for k in range(n):
            input = generateInput(data, data[k][0], data[k][1], data[k][2], data[k][3])
            # Conv2d的规定输入数据格式为(batch, channel, Height, Width)
            train_X = torch.from_numpy(input).view(1, in_channels, height, width).float()
            train_Y = torch.from_numpy(np.array(data[k][4])).view(1, -1).float()
            var_x = Variable(train_X)
            var_y = Variable(train_Y)

            # 前向传播
            out = model(var_x)
            var_result = out.matmul(v0)
            loss = criterion(var_result, var_y)
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

        if (e + 1) % 100 == 0:  # 每 n 次输出结果
            print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
            # outputPath = outputDirectory + "\\States_" + now + "_checkpoint" + str(int((e + 1) / 100)) + ".pth"
            # save(outputPath, model)

            # 下面是eval的内容
            # model = model.eval()
            # test_pred = model(test_X)
            # calAccuracy(test_Y, test_pred)


    # 定义函数
    def Yst(data, x, y, z, t):
        input = generateInput(data, x, y, z, t)
        input = torch.from_numpy(input).view(1, in_channels, height, width).float()
        out = model(var_x)
        yst = out.matmul(v0)
        return yst.reshape(-1).detach().numpy()[0]

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
# 得到逆矩阵
def matInverse(K):
    try:
        K_inverse = np.linalg.inv(K)
    except:
        print(K)
        print("矩阵不存在逆矩阵")
        print(exit(0))
    return K_inverse


# 计算矩阵K
def calMatrixK(data, Yst):
    K = []
    n = len(data)
    for i in range(n):
        templst = []
        for j in range(n):
            ds = calSDis(data[i][0], data[i][1], data[j][0], data[j][1])
            dp = calTDis(data[i][2], data[j][2])
            dt = calTDis(data[i][3], data[j][3])
            yst = Yst(data[i][0], data[i][1], data[j][0], data[j][1],
                      data[i][3], data[j][3], ds, dt)
            templst.append(yst)
        K.append(templst)

    return np.array(K)


# 计算任意点的向量D
def calMaxtrixD(data, Yst, x, y, p, t):
    D = []
    for row in data:
        ds = calSDis(row[0], row[1], x, y)
        dp = calTDis(row[2], p)
        dt = calTDis(row[3], t)
        yst = Yst(row[0], row[1], x, y,
                  row[3], t, ds, dt)
        D.append([yst])

    return np.array(D)


####################################################################################
# get data boundary
# def getBoundary(data):
#     x0 = np.array(data)[:, 0]
#     y0 = np.array(data)[:, 1]
#     z0 = np.array(data)[:, 2]
#     t0 = np.array(data)[:, 3]
#
#     return min(x0), max(x0), min(y0), max(y0)


# trans str to double
def transDataStr2Float(data):
    fdata = np.array([[float(col) for col in row] for row in data])
    lstT = fdata[:, 3] - firstDay + 1
    fdata[:, 3] = lstT

    return fdata


def drawDataT(dataT):
    dataT = np.array(dataT)
    t = np.array(dataT[:, 3])
    v = np.array(dataT[:, 4])

    plt.plot(t, v, color='#FFA500', label='Origin Data')
    plt.show()

    return None


def SpatioTemporalNeuralNetworkKriging(data, columns, depth, day):
    outputData, outputColumns = [], ['x', 'y', 'v']

    # 数据清洗，转换时间
    data = transDataStr2Float(data)
    print("Data  Length: ", len(data))

    # 建立模型
    Yst = createYst(data)  # 变异函数 计算Yst

    # 计算矩阵K和逆矩阵K_inverse
    # K = calMatrixK(data, Yst)
    # K_inverse = matInverse(K)

    # 建立网格场，用神经网络和克里金估算每个点的值
    minX, maxX, minY, maxY = getBoundary()
    # v0 = np.array(data)[:, 4]
    print("Boundary: ", minX, maxX, minY, maxY)
    for m in np.arange(minX, maxX, 2):
        for n in np.arange(minY, maxY, 2):
            # D = calMaxtrixD(data, Yst, m, n, depth, day)
            # coef = np.dot(K_inverse, D)
            # v = np.dot(v0, coef)[0]
            v = Yst(data, m, n, depth, day)

            outputData.append([m, n, v])
            print(m, n, v)

    return outputData, outputColumns
