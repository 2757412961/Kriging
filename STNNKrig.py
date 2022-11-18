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


# 求取散点的半方差
def Semivariogram(data):
    dicS, dicT = {}, {}
    DisS, DisT, SemiS, SemiT = [], [], [], []
    n = len(data)

    for i in range(n):
        for j in range(i + 1, n):
            ds = int(calSDis(data[i][0], data[i][1], data[j][0], data[j][1]))
            se = calSemi(data[i][4], data[j][4])
            if ds < 55:
                if ds in dicS.keys():
                    dicS[ds].append(se)
                else:
                    dicS[ds] = [se]

    meanT = np.mean(np.array(data)[:, 3])
    meanSemi = np.mean(np.array(data)[:, 4])
    for i in range(n):
        dt = int(calTDis(data[i][3], meanT))
        se = calSemi(data[i][4], meanSemi)
        if dt in dicT.keys():
            dicT[dt].append(se)
        else:
            dicT[dt] = []

    # dicS[40.5] = [40-1.5]
    # for key in dicS:
    #     DisS.append(key)
    #     if key < 12:
    #         SemiS.append(np.mean(dicS[key]) + 2)
    #     elif key < 20:
    #         SemiS.append(np.mean(dicS[key]) - 1)
    #     elif key < 25:
    #         SemiS.append(np.mean(dicS[key]) - 2)
    #     elif key < 35:
    #         SemiS.append(np.mean(dicS[key]) - 0.1)
    #     elif key < 40:
    #         SemiS.append(np.mean(dicS[key]) + 0.5)
    #     elif key < 45:
    #         SemiS.append(np.mean(dicS[key]) + 1.5)
    #     elif key < 65:
    #         SemiS.append(np.mean(dicS[key]) + 3.25)
    #     elif key < 80:
    #         SemiS.append(30 + np.random.rand() * 4)
    #     else:
    #         SemiS.append(29.5 + np.random.rand() * 1.5)

    for key in dicS:
        DisS.append(key)
        SemiS.append(np.mean(dicS[key]))

    for key in dicT:
        DisT.append(key)
        SemiT.append(np.mean(dicT[key]))

    return DisS, DisT, SemiS, SemiT


# 创建距离变异函数
def createYs(data):
    n_input = 1
    n_hidden = 12
    n_output = 1
    Ys = Net_Model(n_input, n_hidden, n_output)

    # 获得距离 对应 半方差值
    X, DisT, Y, SemiT = Semivariogram(data)
    # n = len(data)
    # X, Y = [], []
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         ds = calSDis(data[i][0], data[i][1], data[j][0], data[j][1])
    #         se = calTDis(data[i][4], data[j][4])
    #         X.append(ds)
    #         Y.append(se)

    # 按照值平均分割数据为200份
    # X, Y = divideByValue(X, Y, 200)
    train_X, test_X = divideDataset(X, 10)
    train_Y, test_Y = divideDataset(Y, 10)

    # 设置LSTM模型数据类型形状
    train_X = torch.from_numpy(train_X).view(-1, 1, 1).float()
    test_X = torch.from_numpy(test_X).view(-1, 1, 1).float()
    train_Y = torch.from_numpy(train_Y).view(-1, 1, 1).float()
    test_Y = torch.from_numpy(test_Y).view(-1, 1, 1).float()

    # 训练模型
    Ys, accuracy = train(Ys, train_X, test_X, train_Y, test_Y, 195)

    def Yss(ds):
        ds = ds.reshape(-1).detach().numpy()
        ds = np.piecewise(ds, [ds < 40],
                     [lambda x_plt: x_plt,
                      lambda x_plt: 40])
        result = Ys(torch.from_numpy(ds).view(-1, 1, 1).float())
        return result

    # 输出结果
    drawResult(Yss, train_X, train_Y)

    return Yss, accuracy


# 创建时间变异函数
def createYt(data):
    n_input = 1
    n_hidden = 12
    n_output = 1
    Yt = Net_Model(n_input, n_hidden, n_output)

    # 获得距离 对应 半方差值
    DisS, X, SemiS, Y = Semivariogram(data)
    # n = len(data)
    # X, Y = [], []
    # for i in range(n):
    #     for j in range(i + 1, n):
    #         dt = calTDis(data[i][3], data[j][3])
    #         se = calTDis(data[i][4], data[j][4])
    #         X.append(dt)
    #         Y.append(se)

    # 按照值平均分割数据为200份
    # X, Y = divideByValue(X, Y, 200)
    train_X, test_X = divideDataset(X, 10)
    train_Y, test_Y = divideDataset(Y, 10)

    # 设置LSTM模型数据类型形状
    train_X = torch.from_numpy(train_X).view(-1, 1, 1).float()
    test_X = torch.from_numpy(test_X).view(-1, 1, 1).float()
    train_Y = torch.from_numpy(train_Y).view(-1, 1, 1).float()
    test_Y = torch.from_numpy(test_Y).view(-1, 1, 1).float()

    # 训练模型
    Yt, accuracy = train(Yt, train_X, test_X, train_Y, test_Y, 600)

    # 输出结果
    drawResult(Yt, train_X, train_Y)

    return Yt, accuracy

def createYt0(data):
    input_size = 3
    hidden_size = 12
    num_layer = 3
    output_size = 1
    Yt = LSTM_Model(input_size, hidden_size, num_layer, output_size)

    # 按照时间排序
    data = sortByIndex(data)

    # 获得距离 对应 半方差值
    n = len(data)
    X, Y = [], []
    for i in range(n):
        for j in range(i + 1, n):
            ds = calSDis(data[i][0], data[i][1], data[j][0], data[j][1])
            dp = calTDis(data[i][2], data[j][2])
            dt = calTDis(data[i][3], data[j][3])
            se = calTDis(data[i][4], data[j][4])
            X.append([ds, dp, dt])
            Y.append(se)

    # 按照值平均分割数据为200份
    # X, Y = divideByValue(X, Y, 200)
    print(len(X))
    X, Y = X[:3000], Y[:3000]
    train_X, test_X = divideDataset(X, 10)
    train_Y, test_Y = divideDataset(Y, 10)

    # 设置LSTM模型数据类型形状
    train_X = torch.from_numpy(train_X).view(-1, 1, 3).float()
    test_X = torch.from_numpy(test_X).view(-1, 1, 3).float()
    train_Y = torch.from_numpy(train_Y).view(-1, 1, 1).float()
    test_Y = torch.from_numpy(test_Y).view(-1, 1, 1).float()

    # 训练模型
    Yt, accuracy = train(Yt, train_X, test_X, train_Y, test_Y, 500)

    # 输出结果
    drawResultYt(Yt, train_X, train_Y)

    return Yt, accuracy


# 导出时空变异函数
def createYst(Ys, Yt):
    Cs0, Ct0, Cst0 = \
        Ys(torch.from_numpy(np.array([100])).view(-1, 1, 1).float()).reshape(-1).detach().numpy()[0], \
        Yt(torch.from_numpy(np.array([100])).view(-1, 1, 1).float()).reshape(-1).detach().numpy()[0], \
        20
    print(Cs0, Ct0, Cst0)

    def Yst(ds, dp, dt):
        ys = Ys(torch.from_numpy(np.array([ds])).view(-1, 1, 1).float())
        ys = ys.reshape(-1).detach().numpy()[0]
        # yt = Yt(torch.from_numpy(np.array([dt])).view(-1, 1, 1).float())
        # yt = yt.reshape(-1).detach().numpy()[0]

        return ys #+ yt - (Cs0 + Ct0 - Cst0) / (Cs0 * Ct0) * ys * yt

    return Yst


# 绘制结果
def drawResult(model, X, Y):
    plt.title('变异函数拟合曲线')

    x0 = X.reshape(-1).detach().numpy()
    y0 = Y.reshape(-1).detach().numpy()
    plt.scatter(x0, y0, label='原始点')

    x_plt = np.arange(min(x0) * 0.9, max(x0) * 1.1, 0.1)
    y_plt = model(torch.from_numpy(x_plt).view(-1, 1, 1).float())
    y_plt = y_plt.reshape(-1).detach().numpy()
    plt.plot(x_plt, y_plt, color='#FFA500', label='拟合曲线')

    plt.legend()
    plt.show()


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
            yst = Yst(ds, dp, dt)
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
        yst = Yst(ds, dp, dt)
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


def SpatioTemporalNeuralNetworkKriging(dataS, dataT, data, columns, depth, day):
    plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    outputData, outputColumns = [], ['x', 'y', 'v']

    # 数据清洗，转换时间
    dataS = transDataStr2Float(dataS)
    dataT = transDataStr2Float(dataT)
    data = transDataStr2Float(data)
    # drawDataT(dataT)
    print("DataS Length: ", len(dataS))
    print("DataT Length: ", len(dataT))
    print("Data  Length: ", len(data))

    # 建立模型
    Ys, accuracy = createYs(dataS)  # 神经网络 计算Ys
    Yt, accuracy = createYt(dataT)  # LSTM 计算Yt
    Yst = createYst(Ys, Yt)  # 变异函数 使用乘积模型 计算Yst

    # 计算矩阵K和逆矩阵K_inverse
    K = calMatrixK(data, Yst)
    K_inverse = matInverse(K)

    # 建立网格场，用神经网络和克里金估算每个点的值
    minX, maxX, minY, maxY = getBoundary()
    v0 = np.array(data)[:, 4]
    print("Boundary: ", minX, maxX, minY, maxY)
    for m in np.arange(minX, maxX, 2):
        for n in np.arange(minY, maxY, 2):
            D = calMaxtrixD(data, Yst, m, n, depth, day)
            coef = np.dot(K_inverse, D)
            v = np.dot(v0, coef)[0]

            outputData.append([m, n, v])
            # print(m, n, v)

    return outputData, outputColumns

