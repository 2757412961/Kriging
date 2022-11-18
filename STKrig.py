'''
    Name: Spatio-temporal Kriging
    Creation: 2020-02-22
'''

from osgeo import gdal, ogr, osr
import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
from scipy import optimize
from random import sample

# Const Variable
from ConstVariable import *
Yst = lambda x, y: 0

# trans str to double
def transDataStr2Float(data):
    fdata = np.array([[float(col) for col in row] for row in data])
    lstT = fdata[:, 3] - firstDay + 1
    fdata[:, 3] = lstT

    return fdata

# calculate Distance
def calSDis(x1, y1, x2, y2):
    return np.sqrt((x1-x2)**2 + (y1-y2)**2) + 1

def calTDis(t1, t2):
    return np.abs(t1 - t2) + 1

def calSemi(v1, v2):
    return np.power((v1 - v2), 2) / 2

# Divide the list into n blocks
def divideByCount(x0, y0, n):
    n = int(np.ceil(len(x0) / float(n)))
    arr = [[x0[i], y0[i]] for i in range(len(x0))]
    arr = sorted(arr, key=lambda row: (row[0]))
    arr = [arr[i:i+n] for i in range(0, len(arr), n)]

    x0 = [np.mean(np.array(chunk)[:, 0]) for chunk in arr]
    y0 = [np.mean(np.array(chunk)[:, 1]) for chunk in arr]

    return x0, y0

# Divide the list into n blocks
def divideByValue(x0, y0, n):
    minX, maxX, diff = min(x0), max(x0), (max(x0) - min(x0))/n
    tempX, tempY = [[] for i in range(n)], [[] for i in range(n)]
    for i in range(len(x0)):
        for j in range(n):
            if minX + diff*j <= x0[i] < minX + diff*(j+1):
                tempX[j].append(x0[i])
                tempY[j].append(y0[i])
                break

    emptyCount = 0
    for i in range(n):
        if len(tempY[i-emptyCount]) == 0:
            del tempX[i-emptyCount], tempY[i-emptyCount]
            emptyCount += 1

    tempX = [np.mean(x) for x in tempX]
    tempY = [np.mean(y) for y in tempY]

    return tempX, tempY


# 指数模型，变异函数
def YExponential(dis, C0, C, A):
    return np.piecewise(dis, [dis <= 0],
                        [lambda dis: C0,
                         lambda dis: C0 + C * (1 - np.exp(-dis/A))])

# 球状模型，变异函数
def YSpherical(dis, C0, C, A):
    return np.piecewise(dis, [dis <= A],
                        [lambda dis: C0 + C * (3*dis/A - dis**3/A**3) / 2,
                         lambda dis: C0 + C])

# 高斯模型，变异函数
def YGaussian(dis, C0, C, A):
    return np.piecewise(dis, [dis <= 0],
                        [lambda dis: C0,
                         lambda dis: C0 + C * (1 - np.exp(-dis**2/A**2))])

# 孔穴效应，变异函数
def YHoleEffect(dis, C0, C, A):
    return np.piecewise(dis, [dis <= 0],
                        [lambda dis: C0,
                         lambda dis: C0 + C * (1 - np.exp(-dis/A) * np.cos(dis/A))])


# 拟合单个变异函数
def simulateCurve(x0, y0, Yn):
    # 绘制原始数据
    plt.title('变异函数拟合曲线')
    plt.scatter(x0, y0, label='原始点')

    # 将数据合并为十个点
    # x0, y0 = divideByValue(x0, y0, 20)
    # plt.scatter(x0, y0, color='#FFA500', marker='+', label='MeanPoint')

    # 拟合曲线，返回曲线参数
    # np.vectorize(Yn)
    fit_params, pcov = optimize.curve_fit(Yn, x0, y0)

    # 绘制拟合好的曲线
    x_plt = np.arange(min(x0)*0.9, max(x0) * 1.1, 0.1)
    y_plt = Yn(x_plt, fit_params[0], fit_params[1], fit_params[2])
    # p = np.polyfit(x0, y0, 4)
    # y_plt = np.polyval(p, x_plt)
    plt.plot(x_plt, y_plt, color='#FFA500', label='拟合曲线')
    plt.legend() # 绘制图例
    plt.show()

    return fit_params

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
        # DisT.append(dt)
        # SemiT.append(se)

    for key in dicS:
        DisS.append(key)
        SemiS.append(np.mean(dicS[key]))

    for key in dicT:
        DisT.append(key)
        SemiT.append(np.mean(dicT[key]))

    return DisS, DisT, SemiS, SemiT


# 导出时空变异函数
def createYst(data):
    DisS, DisT, SemiS, SemiT = Semivariogram(data)

    Ys_para = simulateCurve(DisS, SemiS, YGaussian)
    print('Ys_para: ' + str(Ys_para))
    Yt_para = simulateCurve(DisT, SemiT, YHoleEffect) # YHoleEffect
    print('Yt_para: ' + str(Yt_para))

    # calculate Yst
    Cs0, Ct0, Cst0 = Ys_para[0]+Ys_para[1], Yt_para[0]+Yt_para[1], max(SemiS)
    print("Cs0, Ct0, Cst0: ", Cs0, Ct0, Cst0)
    k1 = (Cs0 + Ct0 - Cst0) / (Cs0 * Ct0)
    k2 = (Cst0 - Ct0) / Cs0
    k3 = (Cst0 - Cs0) / Ct0

    def Yst(disS, disT):
        ys = YGaussian(disS, Ys_para[0], Ys_para[1], Ys_para[2])
        yt = YHoleEffect(disT, Yt_para[0], Yt_para[1], Yt_para[2]) # YHoleEffect
        return ys #+ yt - k1*ys*yt

    return Yst


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
def calMatrixK(data):
    K = []
    n = len(data)
    for i in range(n):
        templst = []
        for j in range(n):
            ds = calSDis(data[i][0], data[i][1], data[j][0], data[j][1])
            dt = calTDis(data[i][3], data[j][3])
            yst = Yst(ds, dt)
            templst.append(yst)
        K.append(templst)

    return np.array(K)

# 计算任意点的向量D
def calMaxtrixD(data, x, y, t):
    D = []
    for row in data:
        ds = calSDis(row[0], row[1], x, y)
        dt = calTDis(row[3], t)
        yst = Yst(ds, dt)
        D.append([yst])

    return np.array(D)


def SpatioTemporalKriging(data, columns, depth, day):
    plt.rcParams['font.sans-serif']=['SimHei'] #显示中文标签
    global Yst # global声明
    outputData, outputColumns = [], ['x', 'y', 'v']

    data = transDataStr2Float(data)
    print("Data Length: ", len(data))

    Yst = createYst(data)
    K = calMatrixK(data)
    K_inverse = matInverse(K)

    minX, maxX, minY, maxY = getBoundary()
    v0 = np.array(data)[:, 4]
    print("Boundary: ", minX, maxX, minY, maxY)
    for m in np.arange(minX, maxX, 2):
        for n in np.arange(minY, maxY, 2):
            D = calMaxtrixD(data, m, n, day)
            coef = np.dot(K_inverse, D)
            v = np.dot(v0, coef)[0]

            outputData.append([m, n, v])
            # print([m, n, v])


    return outputData, outputColumns
