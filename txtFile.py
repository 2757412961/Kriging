import numpy as np

fileDir = "F:\\EnglishPath\\3ArcGIS"
fileName = "Export_Output.txt"

filePath = fileDir + "\\" + fileName
file = open(filePath, encoding='gbk')

dic = {}
a = file.readline().split(',')
for i in range(len(a)):
    dic[a[i]] = i
print(dic)

lst = []
def readTxtFile():
    line = file.readline()
    while line:
        value = line.split(',')
        templst = [int(value[0]),
                   float(value[3]),
                   float(value[4]),
                   float(value[2])]
        lst.append(templst)
        line = file.readline()

    return lst

