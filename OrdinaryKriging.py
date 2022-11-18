import math
import numpy as np
import matplotlib.pyplot as plt

from scipy import optimize

fit_params = 0

def cal2DDis(x1, y1, x2, y2):
    # ArcGIS 中的几何计算是平面的 - 也就是说，在投影空间中计算而不是在球空间或测地线空间中计算
    # return np.arccos(np.cos(y1)*np.cos(y2)*np.cos(x1-x2)+np.sin(y1)*np.sin(y2)) # view as angle
    return math.sqrt((x1-x2)**2 + (y1-y2)**2) # view as digit

# inputs => [['id', 'x', 'y', 'v']]
def calSemiVariogram(inputs):
    svPoints = []
    for i in range(len(inputs)):
        for j in range(i+1, len(inputs)):
            dis = cal2DDis(inputs[i][1], inputs[i][2], inputs[j][1], inputs[j][2])
            value = (inputs[i][3]-inputs[j][3]) ** 2 / 2
            svPoints.append([dis, value])

    # plt.title('inputs')
    # plt.scatter([i[1] for i in inputs], [i[2] for i in inputs], alpha=0.8)
    # plt.show()
    return svPoints

def funExponential(x, a, b, c):
    return a + b * (1 - np.exp(-x/c))

def calCij(x):

    return funExponential(x, fit_params[0], fit_params[1], fit_params[2])

def simSemiVariggram(svPoints):
    x0 = np.array([i[0] for i in svPoints])
    y0 = np.array([i[1] for i in svPoints])
    plt.title('svPoints')
    plt.scatter(x0, y0)
    # plt.plot(x0, y0)
    # plt.show()
    np.vectorize(funExponential)
    fit_params, pcov = optimize.curve_fit(funExponential, x0, y0)

    x_plt = np.arange(0, max(x0)*1.1, 0.1)
    y_plt = funExponential(x_plt, fit_params[0], fit_params[1], fit_params[2])
    plt.plot(x_plt, y_plt)
    plt.show()
    return fit_params

def calMatrixK(inputs, fit_params):
    K = []
    for i in range(len(inputs)):
        templst = []
        for j in range(len(inputs)):
            dis = cal2DDis(inputs[i][1], inputs[i][2], inputs[j][1], inputs[j][2])
            val = funExponential(dis, fit_params[0], fit_params[1], fit_params[2])
            templst.append(val)
        K.append(templst)

    return np.array(K)

def calMaxtrixD(inputs, m, n):
    D = []
    for i in range(len(inputs)):
        dis = cal2DDis(inputs[i][1], inputs[i][2], m, n)
        D.append([calCij(dis)])

    return np.array(D)

def kring(inputs):
    svPoints = calSemiVariogram(inputs)
    print(svPoints)
    global fit_params
    fit_params= simSemiVariggram(svPoints)
    print(fit_params)
    K = calMatrixK(inputs, fit_params)
    K_inverse  = np.linalg.inv(K)
    print(K_inverse)

    x0 = np.array([i[1] for i in inputs])
    y0 = np.array([i[2] for i in inputs])
    v0 = np.array([i[3] for i in inputs])
    minx = min(x0)
    miny = min(y0)
    maxx = max(x0)
    maxy = max(y0)

    data = []

    for m in np.arange(minx, maxx, 0.5):
        for n in np.arange(miny, maxy, 0.5):
            D = calMaxtrixD(inputs, m, n)
            lam = np.dot(K_inverse, D)
            v = np.dot(v0, lam)
            a = math.fabs(v)
            if v < 0 or v > 1100:
                print(v)

            plt.title('result')
            plt.scatter([m], [n], c='b', alpha=a/1100)

            # 20200219
            dic = {}
            dic['x'] = m; dic['y'] = n; dic['v'] = v
            data.append(dic)

    plt.show()

    return 0