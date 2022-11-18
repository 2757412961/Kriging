'''
    Name: NetWork Tools
    Creation: 2020-03-09
'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from pytorchtools import EarlyStopping

import numpy as np
import matplotlib.pyplot as plt

# Const Variable
from ConstVariable import *


# divide dataset as train and test
def divideDataset(data_, ratio):
    train, test = [], []
    for i in range(len(data_)):
        if i % ratio != 0:
            train.append(data_[i])
        else:
            test.append(data_[i])
    train = np.array(train)
    test = np.array(test)

    return train, test


# 训练模型
def train(model, train_X, test_X, train_Y, test_Y, epcho):
    import Utils
    now = Utils.getNow()
    # criterion = nn.MSELoss()
    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    accs = []
    trai_losses = []
    test_losses = []
    # earlystopping = EarlyStopping(patience=25)
    # 开始训练
    for e in range(epcho):
        var_x = Variable(train_X)
        var_y = Variable(train_Y)
        # 前向传播
        out = model(var_x)
        loss = criterion(out, var_y)
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))

        #######################################################################################
        test_pred = model(test_X)
        accs.append(calAccuracy(test_Y, test_pred))
        loss_vaild = criterion(test_pred, test_Y)

        trai_losses.append(loss.item())
        test_losses.append(loss_vaild.item())
        #######################################################################################

        if (e + 1) % 100 == 0:  # 每 n 次输出结果
            print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
            # outputPath = outputDirectory + "\\States_" + now + "_checkpoint" + str(int((e + 1) / 100)) + ".pth"
            # save(outputPath, model)

            # 下面是eval的内容
            model = model.eval()
            test_pred = model(test_X)
            calAccuracy(test_Y, test_pred)
            # drawPred(test_X, test_Y, test_pred)

        # Early Stop
        # earlystopping(loss_vaild.item(), model)
        # if earlystopping.early_stop:
        #     break

    #######
    train_pred = model(train_X)
    accuracy = calAccuracy(train_Y, train_pred)
    # drawPred(test_X, test_Y, test_pred)
    #######################################################################################
    plt.plot(accs)
    plt.show()
    plt.title('模型训练 Loss 图')
    plt.plot(trai_losses, label='train data Loss')
    plt.plot(test_losses, label='test data Loss')
    plt.legend()
    plt.show()
    #######################################################################################
    #########

    return model, accuracy


# 温差在1.5度上下为正确的结果
def calAccuracy(test_Y, test_pred):
    data_y = test_Y.reshape(-1).detach().numpy()
    data_pred = test_pred.reshape(-1).detach().numpy()

    n = len(test_Y)
    correct = 0
    for i in range(n):
        if np.fabs(data_y[i] - data_pred[i]) < 1:
            correct += 1
    print("accuracy is: %.2f%%" % (correct / n * 100))

    return correct / n


def drawPred(test_X, test_Y, test_pred):
    plt_x = test_X[:, 0, 3].detach().numpy()
    plt_y = test_Y.reshape(-1).detach().numpy()
    plt_pred = test_pred.reshape(-1).detach().numpy()

    # 画出实际结果和预测的结果
    plt.title("drawPred")
    plt.scatter(plt_x, plt_pred)
    plt.plot(plt_x, plt_y, 'b', label='real')
    plt.plot(plt_x, plt_pred, 'r', label='prediction')
    plt.legend()
    plt.show()
    plt.close()

    del plt_x, plt_y, plt_pred


def save(savePath, model):
    print("save: " + savePath)
    torch.save(model.state_dict(), savePath)


def load(loadPath, model):
    print("load: " + loadPath)
    model.load_state_dict(torch.load(loadPath))
    model = model.eval()

    return model
