from Utils import *
from ConstVariable import *
import DBUtil
import NNUtil
import STKrig
import STNNKrig
import STNNKrig2
import STNNKrig3
import LSTM

try:
    import gdal
except:
    from osgeo import gdal
    from osgeo.gdalconst import *

# Driver
gdal.AllRegister()
driver = gdal.GetDriverByName('HFA')  # Erdas的栅格数据类型
driver = gdal.GetDriverByName('GTiff')  # Tiff文件

# 打印驱动名称
for idx in range(gdal.GetDriverCount()):
    driver = gdal.GetDriver(idx)
    # print("%10s: %s" % (driver.ShortName, driver.LongName))

#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
#################################################################################
import matplotlib.pyplot as plt

plt.plot(range(100))
plt.show()

exit(0)

#################################################################################
from GeoBiLSTM import *

seq_len = 1013
batch = 1
input_size = 3 + 3 * 2
hidden_size = 64
num_layer = 1
output_size = 1
isbidirectional = True
model = LSTM_Model(input_size, hidden_size, num_layer, output_size, isbidirectional)
print(model)

params = list(model.parameters())
k = 0
for i in params:
    l = 1
    print("该层的结构：" + str(list(i.size())))
    for j in i.size():
        l *= j
    print("该层参数和：" + str(l))
    k = k + l
print("总参数数量和：" + str(k))
exit(0)


#################################################################################
# 2020-04-01
# a = [1, 2, 3, 4, 5, 6]
# a = np.array(a)
# y = lambda x: x+10
# print(a)
# print(y())
#
# exit(0)

#################################################################################
# 2020-03-30
# now = getNow()
# # data, columns = getTemperatureOnlyByXYZT(145, 35, 0, 60, 20, 1, 25440, 15)
# data, columns = getTemperatureOnlyByXYZGT(165, 7, -35, 7, 50, 20, 25380, 150)
# saveAsCsv(outputDirectory + '\\vis\\Checkout_' + now + ".csv", data, columns)
# print(len(data))
#
# exit(0)

#################################################################################
# 2020-03-26
#
# exit(0)

#################################################################################
# 2020-03-23
# from Utils import *
# from ConstVariable import *
# import Testdelete
#
# lon = 145
# dlon = 35
# lat = 0
# dlat = 60
# depth = 250
# juldday = 25440  # 2019-08-27
# day = juldday - firstDay + 1
#
# now = getNow()
# ### step 1
# dataS, columns = getTemperatureByXYZT(lon, dlon, lat, dlat, depth, 6, juldday, 3)
# dataT, columns = getTemperatureByZTGT(depth, 6, juldday, 60)
# data, columns = getTemperatureByXYZT(lon, dlon, lat, dlat, depth, 6, juldday, 15)
# saveAsCsv(outputDirectory + '\\STNNKrig_' + now + "_RAW.csv", data, columns)
# saveAsShp(outputDirectory + '\\STNNKrig_' + now + "_RAW.shp", data, columns)
# ### step 2
# outputData, outputColumns = \
#     Testdelete.SpatioTemporalNeuralNetworkKriging(dataS, dataT, data, columns, depth, day)
# # saveAsCsv(outputDirectory + '\\STNNKrig_' + now + ".csv", outputData, outputColumns)
# saveAsShp(outputDirectory + '\\STNNKrig_' + now + ".shp", outputData, outputColumns)
# ### step 3
# IDW(outputDirectory + '\\STNNKrig_' + now + ".tif",
#     outputDirectory + '\\STNNKrig_' + now + ".shp")
#
#
# print("Now time: " + getNow())
exit(0)

#################################################################################
# 2020-03-19
# from NNUtil import *
# import numpy as np
#
# a = np.array([[2, 3, 4],
#               [3, 3, 3],
#               [4, 4, 4],
#               [5, 5, 5],
#               [6, 6, 6]])
#
# # LSTM模型
# class LSTM_Model(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layer, output_size):
#         super(LSTM_Model, self).__init__()
#         self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
#         self.layer2 = nn.Linear(hidden_size, hidden_size)
#         self.layer3 = nn.Linear(hidden_size, hidden_size)
#         self.layer4 = nn.Linear(hidden_size, output_size)
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):  # torch.Size([seq, batch_size, hidden_size])
#         x, _ = self.layer1(x)
#         s, b, h = x.size()  # torch.Size([826, 1, 12])
#         x = x.view(s * b, h)  # torch.Size([826, 12])
#         x = self.layer2(F.selu(self.dropout(x)))  # torch.Size([826, 12])
#         x = self.layer3(F.relu(self.dropout(x)))  # torch.Size([826, 12])
#         x = self.layer4(F.selu(self.dropout(x)))  # torch.Size([826, 1])
#         x = x.view(s, b, -1)  # torch.Size([826, 1, 1])
#
#         return x
#
#
# n = 3
# input_size = 3
# hidden_size = 12
# num_layer = 2
# output_size = n
#
# model = LSTM_Model(input_size, hidden_size, num_layer, output_size)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
# data = np.random.rand(5 * n)
# # Conv2d的规定输入数据格式为(batch, channel, Height, Width)
# train_X = torch.from_numpy(data).view(1, in_channels, height, width).float()
# train_Y = torch.from_numpy(np.array(3.54)).view(1, -1).float()
#
# var_x = Variable(train_X)
# var_y = Variable(train_Y)
# print(var_y)
# # 前向传播
# out = model(var_x)
# print(out.size())
#
# v0 = torch.from_numpy(np.random.rand(n)).view(-1, 1).float()
# print(v0.size())
#
# v = out.matmul(v0)
# print(v)
#
# loss = criterion(v, var_y)
# print(loss)
# # 反向传播
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
#
# exit(0)
# exit(0)



#################################################################################
# 2020-03-18
# import numpy as np
#
# from NNUtil import *
#
#
# class STNNKrig(nn.Module):
#     def __init__(self, output_size):
#         super(STNNKrig, self).__init__()
#         self.n = output_size
#         self.layer1 = nn.Conv2d(in_channels=1,
#                                 out_channels=3,
#                                 kernel_size=3,
#                                 stride=1,
#                                 padding=1,
#                                 groups=1)
#         self.layer2 = nn.MaxPool2d(kernel_size=2, padding=1)
#         self.layer3 = nn.Linear(3 * (int(n/2)+1) * 3, 2048)
#         self.layer4 = nn.Linear(2048, output_size)
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):  # torch.Size([batch, channel, Height, Width])
#         # print(x.size()) # torch.Size([1, 1, n, 5])
#         x = self.layer1(x) # torch.Size([1, 3, n, 5])
#         x = self.layer2(F.selu(self.dropout(x))) # torch.Size([1, 3, int(n/2)+1, 3])
#         b, c, h, w = x.size()
#         x = x.view(b, c * h * w) # torch.Size([1, c * h * w])
#         x = self.layer3(F.selu(self.dropout(x))) # torch.Size([1, 2048])
#         x = self.layer4(F.relu(self.dropout(x))) # torch.Size([1, n])
#         x = x.view(b, output_size)
#
#         return x
#
#
# n = 1032
# in_channels = 1
# out_channels = 3
# width = 5
# height = n
# output_size = n
#
# model = STNNKrig(output_size)
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
#
# data = np.random.rand(5 * n)
# # Conv2d的规定输入数据格式为(batch, channel, Height, Width)
# train_X = torch.from_numpy(data).view(1, in_channels, height, width).float()
# train_Y = torch.from_numpy(np.array(3.54)).view(1, -1).float()
#
# var_x = Variable(train_X)
# var_y = Variable(train_Y)
# print(var_y)
# # 前向传播
# out = model(var_x)
# print(out.size())
#
# v0 = torch.from_numpy(np.random.rand(n)).view(-1, 1).float()
# print(v0.size())
#
# v = out.matmul(v0)
# print(v)
#
# loss = criterion(v, var_y)
# print(loss)
# # 反向传播
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
#
# exit(0)

#################################################################################
# 2020-03-18
# import torch
# from torch.autograd import Variable
#
# ##单位矩阵来模拟输入
# input = torch.ones(1, 1, 5, 5)
# input = Variable(input)
# print(input.size())
# x = torch.nn.Conv2d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding=1, groups=1)
# out = x(input)
# print(out.size())
# # print(list(x.parameters()))
# exit()


# print(np.random.random())
# import torch
# import torch.nn as nn
# import numpy as np
# from torch.autograd import Variable
#
# x = torch.Tensor([1, 10, 100, 1000, 10000, 100000]).view(1, 2, -1, 1, 1)
# print(x.size())
# # Conv2d的规定输入数据格式为(batch, channel, Height, Width)
# # Conv3d的规定输入数据格式为(batch, channel, Depth, Height, Width)
# x = Variable(x)
#
# conv = nn.Conv3d(in_channels=2,
#                  out_channels=6,
#                  kernel_size=(2, 1, 1),
#                  stride=1,
#                  padding=0,
#                  dilation=1, # 参数dilation的作用为： 控制卷积核元素的间隔大小.
#                  bias=False)
#
#
# print(conv.weight.data.size())
# ## 	conv.weight.data.size()的返回值:
# #		# (num_kernels, num_filters_perkernel, (Depth,) Height, Width)
# # 		#	i.e.:(out_channels, in_channels/group, kernel_size)
#
# output = conv(x)
# # print('output=', output.data)
# print('outputsize=', output.data.size())
# # output.data.size()的返回值：
# # 	(batch, out_channels/ or num_of_featurecube, size_of_featurecube)
#
# exit(0)

#################################################################################
# 2020-03-12
# from Utils import *
# from ConstVariable import *
#
# ExtractAllNc2Csv(geoDirectory, csvDirectory)
# importAllcsv2pgDB(csvDirectory)
#
# now = getNow()
# data, columns = getTemperatureByZT(20, 1, 24270, 5)
# saveAsCsv(outputDirectory + '\\RAW_' + now + ".csv", data, columns)
# saveAsShp(outputDirectory + '\\RAW_' + now + ".shp", data, columns)
#
# exit(0)

#################################################################################
# 2020-03-09
# import torch
# train_X = torch.from_numpy(np.array([[1, 2, 3], [4, 5, 6]])).view(-1, 1, 3).float()
# print(train_X)
# exit(0)
# a = [1, 2, np.NAN, [], 3, np.NAN, 3, [], 9]
# print(a)
# count = 0
# for i in range(len(a)):
#     if a[i-count] == 3:
#         del a[i-count]
#         count += 1
# # for x in a:
# #     if x == np.NAN:
# #         del x
# print(a)

#################################################################################
# 2020-03-06
# print(np.array(np.array([1, 2,3 ])))
# outputPath = "F:\\EnglishPath\\1MyPaper\\3ArcGIS\\a4.tif"
# shapefilePath = "F:\\EnglishPath\\1MyPaper\\3ArcGIS\\LSTM_20200307_084814.shp"
# def IDW(outputPath, shapefilePath, option=None):
#     print("IDW: ", outputPath)
#     option = gdal.GridOptions(format='GTiff',
#                           # width=2500,height=2500,
#                           algorithm='invdistnn:power=2:smoothing=3:'
#                                     'max_points=20:min_points=0:nodata=0.0',
#                           # layers=['myfirst'],
#                           zfield='v'
#                               )
#
#     out = gdal.Grid(outputPath, shapefilePath, options=option)
#
#     return 0
# IDW(outputPath, shapefilePath)
# exit(0)

#################################################################################
# 2020-03-05
# LSTM.run(data, columns)
# outputDirectory = "F:\\EnglishPath\\1MyPaper\\3ArcGIS"
#
# data, columns = LSTM.Grid(200, 25355)
# now = getNow()
# saveAsCsv(outputDirectory + '\\LSTM_' + now + ".csv", data, columns)
# saveAsShp(outputDirectory + '\\LSTM_' + now + ".shp", data, columns)
# IDW(outputDirectory + '\\LSTM_' + now + ".tif", outputDirectory + '\\LSTM_' + now + ".shp")
# exit(0)

#################################################################################
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
#
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layer, output_size):
#         super(LSTMModel, self).__init__()
#         self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
#         self.layer2 = nn.Linear(hidden_size, hidden_size)
#         self.layer3 = nn.Linear(hidden_size, hidden_size)
#         self.layer4 = nn.Linear(hidden_size, output_size)
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#         x, _ = self.layer1(x)
#         s, b, h = x.size()
#         # print(x.size()) # torch.Size([826, 1, 6])
#         x = x.view(s * b, h)
#         # print(x.size()) # torch.Size([826, 6])
#         x = self.layer2(F.selu(self.dropout(x)))
#         # print(x.size()) # torch.Size([826, 6])
#         x = self.layer3(F.relu(self.dropout(x)))
#         # print(x.size()) # torch.Size([826, 1])
#         x = self.layer4(F.selu(self.dropout(x)))
#         x = x.view(s, b, -1)
#         # print(x.size()) # torch.Size([826, 1, 1])
#
#         return x
#
# # get data boundary
# def getBoundary(data):
#     x0 = np.array(data)[:, 0]
#     y0 = np.array(data)[:, 1]
#     z0 = np.array(data)[:, 2]
#     t0 = np.array(data)[:, 3]
#
#     return min(x0), max(x0), min(y0), max(y0), np.mean(z0), np.mean(t0)
#
#
# def output(data):
#     sss = LSTMModel(4, 18, 2, 1)
#     sss.load_state_dict(torch.load("F:\\EnglishPath\\1MyPaper\\3ArcGIS\\States_20200304_141226.pth"))
#     sss.eval()
#
#     minX, maxX, minY, maxY, meanZ, meanT = getBoundary(data)
#     v0 = np.array(data)[:, 4]
#     print("Boundary: ", minX, maxX, minY, maxY)
#     outputData, outputColumns = [], ['x', 'y', 'v']
#     for x in np.arange(minX, maxX, 1):
#         for y in np.arange(minY, maxY, 1):
#             v = sss(torch.tensor([x, y, meanZ, meanT]).view(-1, 1, 4))
#             v = v.view(-1).detach().numpy()[0]
#             outputData.append([x, y, v])
#
#     return outputData, outputColumns
#
#
# #################################################################################
# # 2020-03-03
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# from torch.autograd import Variable
#
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
#
# outputDirectory = "F:\\EnglishPath\\1MyPaper\\3ArcGIS"
# csvFilePath = 'F:\\EnglishPath\\1MyPaper\\3ArcGIS\\RAW_20200305_205834.csv'
#
#
# # 数据预处理
# data_csv = pd.read_csv(csvFilePath, header=None)
# data_csv = data_csv.dropna()  # 滤除缺失数据
# dataset = data_csv.values   # 获得csv的值
# dataset = dataset.astype('float32')
# data = np.array(dataset)[:, 1:]
# print(data)
# print(len(data))
#
# # 归一化 省略
# max_value = np.max(dataset)  # 获得最大值
# min_value = np.min(dataset)  # 获得最小值
# scalar = max_value - min_value  # 获得间隔数量
# dataset = list(map(lambda x: x / scalar, dataset)) # 归一化
#
# # 按时间排序
# data = sorted(data, key=lambda x: x[3])
# data = np.array(data)
# tempAllT = data[:, 3]
# minT = np.min(tempAllT)
# tempAllT = tempAllT - minT + 1
# data[:, 3] = tempAllT
# print(data)
#
# # 设置数据集
# data_X = data[:, :4]
# data_Y = data[:,  4]
#
# # 设置训练集和测试集
# # train_size = int(len(data_X) * 0.8)
# # test_size = len(data_X) - train_size
# # train_X = data_X[:train_size]
# # train_Y = data_Y[:train_size]
# # test_X = data_X[train_size:]
# # test_Y = data_Y[train_size:]
# train_X, train_Y, test_X, test_Y = [], [], [], []
# for i in range(len(data_X)):
#     if i % 10 != 0:
#         train_X.append(data_X[i])
#         train_Y.append(data_Y[i])
#     else:
#         test_X.append(data_X[i])
#         test_Y.append(data_Y[i])
# train_X, train_Y, test_X, test_Y = np.array(train_X), np.array(train_Y), np.array(test_X), np.array(test_Y)
#
# # 设置LSTM模型数据类型形状
# train_X = torch.from_numpy(train_X).view(-1, 1, 4)
# train_Y = torch.from_numpy(train_Y).view(-1, 1, 1)
# test_X  = torch.from_numpy(test_X) .view(-1, 1, 4)
# test_Y  = torch.from_numpy(test_Y) .view(-1, 1, 1)
#
# # temp insert-----------------------------------------
# # outputData, outputColumns = output(data)
# # now = getNow()
# # saveAsCsv(outputDirectory + '\\Out_' + now + ".csv", outputData, outputColumns)
# # saveAsShp(outputDirectory + '\\Out_' + now + ".shp", outputData, outputColumns)
# # IDW(outputDirectory + '\\Out_' + now + ".tif", outputDirectory + '\\Out_' + now + ".shp")
# # exit(0)
#
#
# # 建立LSTM模型
# class LSTMModel(nn.Module):
#     def __init__(self, input_size, hidden_size, num_layer, output_size):
#         super(LSTMModel, self).__init__()
#         self.layer1 = nn.LSTM(input_size, hidden_size, num_layer)
#         self.layer2 = nn.Linear(hidden_size, hidden_size)
#         self.layer3 = nn.Linear(hidden_size, hidden_size)
#         self.layer4 = nn.Linear(hidden_size, output_size)
#         self.dropout = nn.Dropout(0.5)
#
#     def forward(self, x):
#         x, _ = self.layer1(x)
#         s, b, h = x.size()
#         # print(x.size()) # torch.Size([826, 1, 6])
#         x = x.view(s * b, h)
#         # print(x.size()) # torch.Size([826, 6])
#         x = self.layer2(F.selu(self.dropout(x)))
#         # print(x.size()) # torch.Size([826, 6])
#         x = self.layer3(F.relu(self.dropout(x)))
#         # print(x.size()) # torch.Size([826, 1])
#         x = self.layer4(F.selu(self.dropout(x)))
#         x = x.view(s, b, -1)
#         # print(x.size()) # torch.Size([826, 1, 1])
#
#         return x
#
#
# lstm = LSTMModel(4, 18, 2, 1)
#
# criterion = nn.MSELoss()
# optimizer = torch.optim.Adam(lstm.parameters(), lr=1e-2)
#
# # 开始训练
# for e in range(1000):
#     var_x = Variable(train_X)
#     var_y = Variable(train_Y)
#     # 前向传播
#     out = lstm(var_x)
#     loss = criterion(out, var_y)
#     # 反向传播
#     optimizer.zero_grad()
#     loss.backward()
#     optimizer.step()
#     print(loss)
#
#     if (e + 1) % 100 == 0:  # 每 100 次输出结果
#         print('Epoch: {}, Loss: {:.5f}'.format(e + 1, loss.item()))
#         now = getNow()
#         outputPath = outputDirectory + '\\States_' + now + ".pth"
#         torch.save(lstm.state_dict(), outputPath)
#
#         model = LSTMModel(4, 18, 2, 1)
#         model.load_state_dict(torch.load(outputPath))
#         model = model.eval()
#
#         pred_test = lstm(test_X)
#         print(pred_test)
#
#         # 画出实际结果和预测的结果
#         plt.title(str(e+1) + outputPath)
#         plt.scatter(test_X[:, 0, 3].detach().numpy(), pred_test.reshape(-1).detach().numpy())
#         plt.plot(test_X[:, 0, 3].detach().numpy(), pred_test.reshape(-1).detach().numpy(), 'r', label='prediction')
#         plt.plot(test_X[:, 0, 3].detach().numpy(), test_Y.reshape(-1).detach().numpy(), 'b', label='real')
#         plt.legend()
#         plt.show()
#
#
# # Print model's state_dict
# print("Model's state_dict:")
# for param_tensor in lstm.state_dict():
#     print(param_tensor, "\t", lstm.state_dict()[param_tensor].size())
# # # Print optimizer's state_dict
# # print("optimizer's state_dict:")
# # for var_name in optimizer.state_dict():
# #     print(var_name,"\t",optimizer.state_dict()[var_name])
#
# now = getNow()
# outputPath = outputDirectory + '\\States_' + now + ".pth"
# torch.save(lstm.state_dict(), outputPath)
#
# lstm = LSTMModel(4, 12, 2, 1)
# lstm.load_state_dict(torch.load(outputPath))
# lstm = lstm.eval()
#
# pred_test = lstm(test_X)
# print(pred_test)
#
# # 画出实际结果和预测的结果
# plt.scatter(test_X[:, 0, 3].detach().numpy(), pred_test.reshape(-1).detach().numpy())
# plt.plot(test_X[:, 0, 3].detach().numpy(), pred_test.reshape(-1).detach().numpy(), 'r', label='prediction')
# plt.plot(test_X[:, 0, 3].detach().numpy(), test_Y.reshape(-1).detach().numpy(), 'b', label='real')
# plt.legend()
# plt.show()
#
# exit(0)


#################################################################################
# 2020-03-02
# data, columns = getTemperatureByZT(200, 100, 25321, 30)
# saveAsCsv("F:\\EnglishPath\\1MyPaper\\3ArcGIS" + "\\All.csv", data, columns)

#################################################################################
# # 球状模型，变异函数
# def YMyFunc(dis, C0, C, A):
#     return np.piecewise(dis, [dis <= A],
#                         [lambda dis: C0 + C * (3*dis/A - dis**3/A**3) / 2,
#                          lambda dis: C0 + C])
#
# print(1 < 2 < 2)
# print([[]]*2)


#################################################################################
# 2020-03-02
# 指数模型，变异函数
# def YExponential(dis, C0, C, A):
#     return np.piecewise(dis, [dis <= 0],
#                         [lambda dis: C0,
#                          lambda dis: C0 + C * (1 - np.exp(-dis/A))])
#
# # 球状模型，变异函数
# def YSpherical(dis, C0, C, A):
#     return np.piecewise(dis, [dis <= A],
#                         [lambda dis: C0 + C * (3*dis/A - dis**3/A**3) / 2,
#                          lambda dis: C0 + C])
#
# # 高斯模型，变异函数
# def YGaussian(dis, C0, C, A):
#     return np.piecewise(dis, [dis <= 0],
#                         [lambda dis: C0,
#                          lambda dis: C0 + C * (1 - np.exp(-dis**2/A**2))])
#
# # 孔穴效应，变异函数
# def YHoleEffect(dis, C0, C, A):
#     return np.piecewise(dis, [dis <= 0],
#                         [lambda dis: C0,
#                          lambda dis: C0 + C * (1 - np.exp(-dis/A) * np.cos(dis/A))])

# from scipy import optimize

# def z(Yn):
#     x0 = [35.89196659823562, 77.78878811579726, 114.50543955211207, 154.91053804324164, 196.07901500977863, 239.71349693954292, 287.8074881518813, 339.91373997397733, 396.79106486790494, 459.62041727157157, 529.718949437807, 608.580801913923, 696.7231455269176, 797.106392018502, 914.3861249148968, 1054.9083221355297, 1228.45264754386, 1456.0467002503606, 1783.469122496816, 2462.7808937528025]
#     y0 = [5.785118921327641, 7.996056491406722, 8.103822964964044, 8.432556633225985, 8.97424560880456, 8.863459683280235, 9.13738467640825, 9.318496188492125, 9.581834006803032, 9.628739344982518, 9.838896943030306, 9.949648919598213, 10.017901182770206, 10.054160489354476, 10.144454693907447, 10.169897028317932, 10.289298313211471, 10.527279254818474, 10.848936780721088, 10.630492924791685]
#     plt.scatter(x0, y0, color='#FFA500', marker='+', label='MeanPoint')
#
#     # 拟合曲线，返回曲线参数
#     # np.vectorize(Yn)
#     # fit_params, pcov = optimize.curve_fit(Yn, x0, y0)
#     p = np.polyfit(x0, y0, 4)
#
#     # 绘制拟合好的曲线
#     x_plt = np.arange(0, max(x0)*1.1, 0.1)
#     y_plt = np.polyval(p, x_plt)
#     # y_plt = Yn(x_plt, fit_params[0], fit_params[1], fit_params[2])
#     plt.plot(x_plt, y_plt, color='#FFA500', label='svCurve')
#     plt.legend() # 绘制图例
#     plt.show()
#
# z(YExponential)


#################################################################################
# 2020-02-21 za
# import os
# csvDirectory = "F:\\EnglishPath\\1MyPaper\\4ETCData"
# csvDirectory = csvDirectory + '\\' + '12' + '\\' + '12' + '\\' + '12'
# if not os.path.exists(csvDirectory):
#     os.makedirs(csvDirectory)
# if not os.path.exists(csvDirectory):
#     os.makedirs(csvDirectory)

# netcdfPath = 'C:\\Users\\Z\\Desktop\\geo\\atlantic_ocean\\2014\\07\\20140712_prof.nc'
# csvDirectory = "F:\\EnglishPath\\1MyPaper\\4ETCData"

# dataset = nc.Dataset(netcdfPath)
# ExtractNc2Csv(netcdfPath, csvDirectory)

# from DBUtil import *
# a = executeSQL("select * from paper")
# print(a)
# exit(2)


#################################################################################
# 2020-02-20 test
# netcdfPath = 'C:\\Users\\Z\\Desktop\\geo\\atlantic_ocean\\2004\\01\\20040101_prof.nc'
# ExtractNc2Csv(netcdfPath)
# ################################################################################
# 2020-02-20 netCDF数据转换为csv
exit(0)
import netCDF4 as nc
import numpy.ma as ma

netcdfPath = 'C:\\Users\\Z\\Desktop\\geo\\atlantic_ocean\\2014\\07\\20140701_prof.nc'
dataset = nc.Dataset(netcdfPath)
ds_vars = dataset.variables

print(dataset)
# print(dir(dataset))
print('dataset.variables---------------------------------------------------------')
# print(dataset.variables)
print('dataset.variables.keys()--------------------------------------------------')
print(dataset.variables.keys())
print('--------------------------------------------------------------------------')
print(ds_vars['DATE_CREATION'])
print('--------------------------------------------------------------------------')
print(ds_vars['DATE_CREATION'].ncattrs())
print('--------------------------------------------------------------------------')
data_creation = [i.decode('utf-8') for i in ds_vars['DATE_CREATION'][:]]
data_creation = ''.join(data_creation)
print('DATE_CREATION: ' + data_creation)
print('--------------------------------------------------------------------------')
platform_number = ds_vars['PLATFORM_NUMBER'][:]
platform_number = [ma.compressed(i) for i in platform_number]
platform_number = [b''.join(i).decode('utf-8') for i in platform_number]
platform_number = np.array(platform_number)
print('PLATFORM_NUMBER: ' + str(platform_number))
# print('--------------------------------------------------------------------------')
# platform_number = ds_vars['PI_NAME'][:]
# print('PI_NAME: ' + str(platform_number))
# print('--------------------------------------------------------------------------')
# station_parameters = ds_vars['STATION_PARAMETERS'][:]
# station_parameters = [ma.compress_cols(i) for i in station_parameters]
# print('STATION_PARAMETERS: ' + str(station_parameters))
# print('--------------------------------------------------------------------------')
# platform_number = ds_vars['DATA_MODE'][:]
# print('DATA_MODE: ' + str(platform_number))
print('--------------------------------------------------------------------------')
juid = ds_vars['JULD'][:]
juid = np.array(juid)
print('JULD: ' + str(juid))
print('--------------------------------------------------------------------------')
latitude = ds_vars['LATITUDE'][:]
latitude = np.array(latitude)
print('LATITUDE: ' + str(latitude))
print('--------------------------------------------------------------------------')
longitude = ds_vars['LONGITUDE'][:]
longitude = np.array(longitude)
print('LONGITUDE: ' + str(longitude))
# print('--------------------------------------------------------------------------')
# platform_number = ds_vars['POSITION_QC'][:]
# print('POSITION_QC: ' + str(platform_number))
# print('--------------------------------------------------------------------------')
# platform_number = ds_vars['POSITIONING_SYSTEM'][:]
# print(platform_number.mask)
# print('POSITIONING_SYSTEM: ' + str(platform_number))
print('--------------------------------------------------------------------------')
platform_number = ds_vars['PRES'][:]
print('PRES: ' + str(platform_number))
print('--------------------------------------------------------------------------')
pres_adjusted = ds_vars['PRES_ADJUSTED'][:]
pres_adjusted = [ma.compressed(i) for i in pres_adjusted]
pres_adjusted = np.array(pres_adjusted)
print('PRES_ADJUSTED: ' + str(pres_adjusted))
print('--------------------------------------------------------------------------')
temp_adjusted = ds_vars['TEMP_ADJUSTED'][:]
temp_adjusted = [ma.compressed(i) for i in temp_adjusted]
temp_adjusted = np.array(temp_adjusted)
print('TEMP_ADJUSTED: ' + str(temp_adjusted))
print('--------------------------------------------------------------------------')
pasl_adjusted = ds_vars['PSAL_ADJUSTED'][:]
pasl_adjusted = [ma.compressed(i) for i in pasl_adjusted]
pasl_adjusted = np.array(pasl_adjusted)
print('PSAL_ADJUSTED: ' + str(pasl_adjusted))
print('--------------------------------------------------------------------------')
parameter = ds_vars['PARAMETER'][:]
parameter = [ma.compressed(i) for i in parameter]
parameter = [b''.join(i).decode('utf-8') for i in parameter]
parameter = np.array(parameter)
print('PARAMETER: ' + str(parameter))
print('--------------------------------------------------------------------------')

#################################################################################
# 2020-02-19 test
# from Utils import *
# data2PointShp('F:\\EnglishPath\\3ArcGIS\\20200219test.shp', [{'x':1, 'y':2, 'v':3}])
# IDW('F:\\EnglishPath\\3ArcGIS\\20200219_6.tif','F:\\EnglishPath\\3ArcGIS\\20200219test.shp')


#################################################################################
# 2020-02-19
# from osgeo import gdal, ogr, osr
#
# shpPath = 'F:\\EnglishPath\\3ArcGIS\\'
# layerName = '20200219_2.shp'
#
# driver = ogr.GetDriverByName('ESRI Shapefile')
# dataSource = driver.CreateDataSource(shpPath + layerName)
#
# srs = osr.SpatialReference()
# srs.ImportFromEPSG(4326)
#
# layer = dataSource.CreateLayer(layerName, srs, ogr.wkbPoint)
#
# field_name = ogr.FieldDefn("Name", ogr.OFTString)
# field_name.SetWidth(14)
# layer.CreateField(field_name)
#
# field_name = ogr.FieldDefn("data", ogr.OFTReal)
# field_name.SetWidth(14)
# layer.CreateField(field_name)
#
# feature = ogr.Feature(layer.GetLayerDefn())
# feature.SetField('Name', "das")
# feature.SetField('data', 1.2)
# wkt = 'Point(6 10)'
# point = ogr.CreateGeometryFromWkt(wkt)
# feature.SetGeometry(point)
# layer.CreateFeature(feature)
#
# feature = ogr.Feature(layer.GetLayerDefn())
# feature.SetField('Name', "56")
# feature.SetField('data', 3)
# wkt = 'Point(1 2)'
# point = ogr.CreateGeometryFromWkt(wkt)
# feature.SetGeometry(point)
# layer.CreateFeature(feature)
#
# feature = None
# dataSource = None


#################################################################################
# 2020-02-18 使用Grid
# from osgeo import gdal, ogr
#
# rasterPath = 'F:\\EnglishPath\\3ArcGIS\\gm_el_v2_01_07.tif'
# vectorPath = 'F:\\EnglishPath\\3ArcGIS\\myfirst.shp'
# outputPath = 'F:\\EnglishPath\\3ArcGIS\\20200218_output_9.tif'
#
# option = gdal.GridOptions(format='GTiff',
#                       # width=2500,height=2500,
#                       algorithm='invdist:power=2',
#                       # layers=['myfirst'],
#                       zfield='y'
#                           )
#
# out = gdal.Grid(outputPath, vectorPath, options=option)


#################################################################################
# 2020-02-16 使用GDAL创建影像
# import osr
# rasterPath = 'F:\\EnglishPath\\3ArcGIS\\gm_el_v2_01_07.tif'
# vectorPath = 'F:\\EnglishPath\\3ArcGIS\\myfirst.shp'
# outputPath = 'F:\\EnglishPath\\3ArcGIS\\20200216_output_7.tif'
#
# dataset = gdal.Open(rasterPath)
# band = dataset.GetRasterBand(1)
# data = band.ReadAsArray(2222, 4444, 3333, 3333)
#
# raster = np.zeros((512, 512))+3
# driver = gdal.GetDriverByName('GTiff')
# # help(driver)
# dst_ds = driver.Create(outputPath, 5120, 5120, 1, gdal.GDT_Int16)
#
# srs = osr.SpatialReference()
# srs.SetUTM(11, 1)
# srs.SetWellKnownGeogCS('NAD27')
#
# # dst_ds.SetProjection(srs.ExportToWkt())
# dst_ds.SetProjection(dataset.GetProjection())
# dst_ds.SetGeoTransform(dataset.GetGeoTransform())
# dst_ds.GetRasterBand(1).WriteArray(data)

## 创建多波段影像的方法
# import osr, gdal
# rasterPath = 'F:\\EnglishPath\\3ArcGIS\\gm_el_v2_01_07.tif'
# vectorPath = 'F:\\EnglishPath\\3ArcGIS\\myfirst.shp'
# outputPath = 'F:\\EnglishPath\\3ArcGIS\\20200216_output_4.tif'
#
# dataset = gdal.Open(rasterPath)
# width  = dataset.RasterXSize
# height = dataset.RasterYSize
# data = dataset.ReadAsArray(0, 0, width, height)
# geoTransform = dataset.GetGeoTransform()
# proj = dataset.GetProjection()
#
# srs = osr.SpatialReference()
# srs.SetUTM(11, 1)
# srs.SetWellKnownGeogCS('NAD27')
#
# driver = gdal.GetDriverByName('GTiff')
# output = driver.Create(outputPath, width, height, 3, options=["INTERLEAVE=PIXEL"])
# output.SetGeoTransform(geoTransform)
# output.SetProjection(srs.ExportToWkt())
#
# output.WriteRaster(0, 0, width, height, data.tostring(), width, height, band_list=[1,2,3])


#################################################################################
# 2020 对真彩色图像与索引图像的处理
# # from helper import info
# rasterPath = 'F:\\EnglishPath\\3ArcGIS\\gm_el_v2_01_07.tif'
# vectorPath = 'F:\\EnglishPath\\3ArcGIS\\myfirst.shp'
# outputPath = 'F:\\EnglishPath\\3ArcGIS\\20200213_output.tif'
# dataset = gdal.Open(rasterPath)
# band = dataset.GetRasterBand(1)
# a = band.GetRasterColorInterpretation()
# print(a)
# print(type(dataset))
# print(type(dataset.ReadAsArray()))

# colormap = band.GetRasterColorTable()
# dir(colormap)
# colormap.GetCount()
# colormap.GetPaletteInterpretation()
# for i in range(colormap.GetCount() - 10, colormap.GetCount()):
#     print("%i:%s" % (i, colormap.GetColorEntry(i)))


#################################################################################
# 2020-02-15 地图代数
# from osgeo import gdalconst
# import numpy
# rasterPath = 'F:\\EnglishPath\\3ArcGIS\\gm_el_v2_01_07.tif'
# vectorPath = 'F:\\EnglishPath\\3ArcGIS\\myfirst.shp'
# outputPath = 'F:\\EnglishPath\\3ArcGIS\\20200213_output.tif'
# dataset = gdal.Open(rasterPath)
# cols = dataset.RasterXSize
# rows = dataset.RasterYSize
# band1 = dataset.GetRasterBand(1)
# band2 = dataset.GetRasterBand(1)
# print(band2.DataType) # 16位整型 gdalconst.GDT_Int16 3
# data1 = band1.ReadAsArray(0, 0, cols, rows).astype(numpy.int16)
# data2 = band2.ReadAsArray(0, 0, cols, rows).astype(numpy.int16)//2
#
# print(data1)
# print('--------------------------------------------------------------------------')
# print(data2)
# print('--------------------------------------------------------------------------')
#
# mask = numpy.not_equal(data1 + data2, 0) # 这个表达式有问题，导致出现除以0的情况
# # ndvi = (data2 - data1)/(data2 + data1)
# ndvi = numpy.choose(mask, (-99, (data2 - data1) / (data2 + data1)))
# print(ndvi)
# print('--------------------------------------------------------------------------')
# help(numpy.choose)


#################################################################################
# 2020-02-15 访问数据集的数据
# rasterPath = 'F:\\EnglishPath\\3ArcGIS\\gm_el_v2_01_07.tif'
# vectorPath = 'F:\\EnglishPath\\3ArcGIS\\myfirst.shp'
# outputPath = 'F:\\EnglishPath\\3ArcGIS\\20200213_output.tif'
# # from helper import info
# dataset = gdal.Open(rasterPath)
# help(dataset.ReadRaster) # ReadRaster() 读取图像数据(以二进制的形式)
# help(dataset.ReadAsArray) # ReadAsArray() 读取图像数据(以数组的形式)
# import array
# from numpy import *
# print(dataset.RasterXSize)
# a = dataset.ReadAsArray(2500, 2500, 5, 3) # ReadAsArray()读出的是numpy的数组
# print(a)
# b = dataset.ReadRaster(2500, 2500, 3, 3)
# print(b)
# print('---------------------------------------------------')
#
# from gdalconst import *
# band1 = dataset.GetRasterBand(1)
# c = band1.ReadAsArray(1000, 5000, 10, 10, 5, 5)
# # c = band1.ReadAsArray(1000, 5000, 10, 10)
# print(c)
# help(band1.ReadAsArray)
# print('---------------------------------------------------')
#
# # 读取栅格数据方式与效率
# from osgeo import gdal
# import time
# dataset = gdal.Open(rasterPath)
# band = dataset.GetRasterBand(1)
# width, height = dataset.RasterXSize, dataset.RasterYSize
# bw, bh = 128, 128
# bxsize = width/bw
# bysize = height/bh
# band.ReadAsArray(0,0,width,height)
# start = time.time()
# band.ReadAsArray(0,0,width,height)
# print (time.time()-start)
# start2 = time.time()
# # for i in range(int(bysize)):
# #     for j in range(int(bxsize)):
# for j in range(int(bxsize)):
#     for i in range(int(bysize)):
#         band.ReadAsArray(bw*j,bh*i,bw,bh)
# print (time.time()-start2)
# print('---------------------------------------------------')


#################################################################################
# 2020-02-14 pythonGIS Exercise
# rasterPath = 'F:\\EnglishPath\\3ArcGIS\\gm_el_v2_01_07.tif'
# vectorPath = 'F:\\EnglishPath\\3ArcGIS\\myfirst.shp'
# outputPath = 'F:\\EnglishPath\\3ArcGIS\\20200213_output.tif'
# dataset = gdal.Open(rasterPath)
# print(dataset.GetMetadata())
# print(dataset.GetDescription())
# print(dataset.GetGeoTransform())
# print("dataset Raster Count: " + str(dataset.RasterCount))
# print("dataset Raster Count: " + str(dataset.GetRasterBand(1)))
# print((dataset.RasterXSize, dataset.RasterYSize))
# print(dir(dataset))
# print('---------------------------------------------------')
#
# transform = dataset.GetGeoTransform()
# proj = dataset.GetProjection()
# print(proj)
#
# band1 = dataset.GetRasterBand(1)
# print(dir(band1))
# print(band1.XSize)
# print(band1.YSize)
# print("DataType: " + str(band1.DataType))
# print(band1.ComputeBandStats())
# print("Sum: " + str(band1.Checksum()))
# print("MinMax: " + str(band1.ComputeRasterMinMax()))
# print(band1.GetNoDataValue())
# print(band1.GetMaximum())
# print(band1.GetMinimum())
# print('---------------------------------------------------')


#################################################################################
# 2020-02-13 python gdal 矢量转栅格
# rasterPath = 'F:\\EnglishPath\\3ArcGIS\\gm_el_v2_01_07.tif'
# vectorPath = 'F:\\EnglishPath\\3ArcGIS\\myfirst.shp'
# outputPath = 'F:\\EnglishPath\\3ArcGIS\\20200213_output.tif'
# field = 'x'
#
# data = gdal.Open(rasterPath, gdal.gdalconst.GA_ReadOnly)
# geo_transform = data.GetGeoTransform()
# x_min = geo_transform[0]
# y_min = geo_transform[3]
# x_res = data.RasterXSize
# y_res = data.RasterYSize
#
# mb_v = ogr.Open(vectorPath)
# mb_l = mb_v.GetLayer()
# pixel_width = geo_transform[1]
# target_ds = gdal.GetDriverByName('GTiff').Create(outputPath, x_res, y_res, 1, gdal.GDT_Byte)
# target_ds.SetGeoTransform((x_min, pixel_width, 0, y_min, 0, -1 * pixel_width))
# band = target_ds.GetRasterBand(1)
# NoData_value = -999
# band.SetNoDataValue(NoData_value)
# band.FlushCache()
# gdal.RasterizeLayer(target_ds, [1], mb_l, options=["ATTRIBUTE=" + field])
# target_ds = None


#################################################################################
# 2020-02-13 gdal中shapefile坐标度转换到栅格米
# def world2Pixel(padTransform, x, y):
#     piexl = padTransform[0] + padTransform[1] * x + padTransform[2] * y
#     line  = padTransform[3] + padTransform[4] * x + padTransform[5] * y
#     return (piexl, line)
#
# rasterPath = 'F:\\EnglishPath\\3ArcGIS\\gm_el_v2_01_07.tif'
# vectorPath = 'F:\\EnglishPath\\3ArcGIS\\New_Shapefile.shp'
#
# dataset = gdal.Open(rasterPath)
#
# driver = ogr.GetDriverByName("ESRI Shapefile")
# dataSource = driver.Open(vectorPath)
#
# layer = dataSource.GetLayer(0)
#
# minX, maxX, minY, maxY = layer.GetExtent()
# print("原边界(坐标系度): ", minX, minY, maxX, maxY)
#
# geoTrans = dataset.GetGeoTransform()
# minX, minY = world2Pixel(geoTrans, minX, minY)
# maxX, maxY = world2Pixel(geoTrans, maxX, maxY)
# print("新边界(坐标系米): ", minX, minY, maxX, maxY)
