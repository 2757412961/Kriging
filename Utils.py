'''
    Name: Data ETL、Build DataBase; Extract and Save Data; IDW
    Creation: 2020-02-20
'''

import DBUtil

import os
import shutil
from osgeo import gdal, ogr, osr
import numpy as np
import numpy.ma as ma
import netCDF4 as nc
import pandas as pd
import datetime
import scipy as sp
import cv2

###################################### 修复中文乱码问题
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rcParams['legend.fontsize'] = 10
plt.rcParams['font.family'] = ['sans-serif']
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False

# #################表示暂时的处理语句，完整工程需要恢复

'''
#  获取当前时间，格式为%Y%m%d_%H%M%S'
'''
def getNow():
    return datetime.datetime.now().strftime('%Y%m%d_%H%M%S')


# ---------------------------------------------------------------------------------------
# Data ETL、Build DataBase
'''
#  将单个netCDF数据转换为csv
#  netcdfPath nc文件的路径
#  ALLcsvDirectory 输出csv文件夹路径
'''
def ExtractNc2Csv(netcdfPath, ALLcsvDirectory):
    print(netcdfPath)
    try:
        time = netcdfPath[netcdfPath.find('_prof.nc') - 8:netcdfPath.find('_prof.nc')]
        year = time[:4]
        month = time[4:6]
        day = time[6:]

        csvDirectory = ALLcsvDirectory + '\\' + year + '\\' + month + '\\' + day
        if not os.path.exists(csvDirectory):
            os.makedirs(csvDirectory)

        dataset = nc.Dataset(netcdfPath)
        ds_vars = dataset.variables

        platform_number = ds_vars['PLATFORM_NUMBER'][:]
        platform_number = [ma.compressed(i) for i in platform_number]
        platform_number = [b''.join(i).decode('utf-8') for i in platform_number]
        platform_number = np.array(platform_number)
        # print('PLATFORM_NUMBER: ' + str(platform_number))

        latitude = ds_vars['LATITUDE'][:]
        latitude = np.array(latitude)
        # print('LATITUDE: ' + str(latitude))

        longitude = ds_vars['LONGITUDE'][:]
        longitude = np.array(longitude)
        # print('LONGITUDE: ' + str(longitude))

        juld = ds_vars['JULD'][:]
        juld = np.array(juld)
        # print('JULD: ' + str(juld))

        parameter = ds_vars['PARAMETER'][:]
        parameter = [ma.compressed(i) for i in parameter]
        parameter = [b''.join(i).decode('utf-8') for i in parameter]
        parameter = np.array(parameter)
        # print('PARAMETER: ' + str(parameter))

        pres_adjusted = ds_vars['PRES_ADJUSTED'][:]
        pres_adjusted = [ma.compressed(i) for i in pres_adjusted]
        pres_adjusted = np.array(pres_adjusted)
        # print('PRES_ADJUSTED: ' + str(pres_adjusted))

        temp_adjusted = ds_vars['TEMP_ADJUSTED'][:]
        temp_adjusted = [ma.compressed(i) for i in temp_adjusted]
        temp_adjusted = np.array(temp_adjusted)
        # print('TEMP_ADJUSTED: ' + str(temp_adjusted))

        if 'PSAL_ADJUSTED' in ds_vars.keys():
            pasl_adjusted = ds_vars['PSAL_ADJUSTED'][:]
            pasl_adjusted = [ma.compressed(i) for i in pasl_adjusted]
            pasl_adjusted = np.array(pasl_adjusted)
            # print('PSAL_ADJUSTED: ' + str(pasl_adjusted))
        else:
            return -1

        for i in range(len(platform_number)):
            csvOutputPath = csvDirectory + '\\' + str(i) + '_' + platform_number[i] + ".csv"
            if os.path.exists(csvOutputPath):
                continue

            if parameter[i].find("PRES") == -1 or \
                    parameter[i].find("TEMP") == -1 or \
                    parameter[i].find("PSAL") == -1: continue

            if len(pres_adjusted[i]) != len(temp_adjusted[i]) or \
                    len(pres_adjusted[i]) != len(pasl_adjusted[i]) or \
                    len(temp_adjusted[i]) != len(pasl_adjusted[i]): continue

            count = len(pres_adjusted[i])
            data = []
            data.append(np.array([longitude[i]] * count))
            data.append(np.array([latitude[i]] * count))
            data.append(pres_adjusted[i])
            data.append(np.array([int(juld[i])] * count))
            data.append(temp_adjusted[i])
            data.append(pasl_adjusted[i])
            data.append(np.array([str(time)] * count))
            data = np.array(data)
            data = data.T
            columns = ['x', 'y', 'z', 't', 'temp', 'pasl', 'date']

            df = pd.DataFrame(columns=columns, data=data)
            df.to_csv(csvOutputPath)
    except OSError:
        os.remove(netcdfPath)

    return 0


'''
#  将所有netCDF数据转换为csv, 2000~2019
#  geoDirectory 存放geo数据文件夹路径
#  ALLcsvDirectory 输出csv文件夹路径
'''
def ExtractAllNc2Csv(geoDirectory, ALLcsvDirectory):
    years = [str(i) for i in range(2000, 2020)]
    for y in years:
        months = os.listdir(geoDirectory + '\\' + y)
        for m in months:
            netcdfFileNames = os.listdir(geoDirectory + '\\' + y + '\\' + m)
            for ncName in netcdfFileNames:
                if ncName.find("_prof.nc") == -1:
                    shutil.rmtree(geoDirectory + '\\' + y + '\\' + m + '\\' + ncName)
                    continue

                ExtractNc2Csv(geoDirectory + '\\' + y + '\\' + m + '\\' + ncName, ALLcsvDirectory)

    return 0


'''
#  将单个浮标的csv导入数据库
#  csvPath 文件路径
'''
def importcsv2pgDB(csvPath):
    print(csvPath)
    try:
        conn, cur = DBUtil.connect()

        with open(csvPath, 'r', encoding='utf-8', newline='') as f:
            insert_SQL = """COPY paper FROM STDIN WITH (FORMAT CSV, HEADER true, NULL 'nan', DELIMITER ',')"""
            cur.copy_expert(insert_SQL, f)
            conn.commit()
    except Exception as e:
        print(e)
        return -1
    finally:
        cur.close()
        conn.close()

    return 0


'''
#  将所有浮标的csv导入数据库
#  csvDirectory 文件夹路径
'''
def importAllcsv2pgDB(csvDirectory):
    # years = os.listdir(csvDirectory) ##############################
    years = ['2019']
    for y in years:
        months = os.listdir(csvDirectory + '\\' + y)
        for m in months:
            days = os.listdir(csvDirectory + '\\' + y + '\\' + m)
            for d in days:
                csvFiles = os.listdir(csvDirectory + '\\' + y + '\\' + m + '\\' + d)
                for fileName in csvFiles:
                    csvPath = csvDirectory +'\\'+ y +'\\'+ m +'\\'+ d +'\\'+ fileName
                    importcsv2pgDB(csvPath)

    return 0


# ---------------------------------------------------------------------------------------
# Extract and Save Data
'''
#  从数据库中选择合适的数据
#  z、t 值
#  deltaZ、deltaT 上下浮动区间
#  以[[x, y, z, t, temp]]输出
'''
def getTemperatureByZT(z, deltaZ, t, deltaT):
    columns = ['x', 'y', 'z', 't', 'temp']
    sql = "select x, y, avg(z), t, avg(temp) " \
          "from paper " \
          "where z between " + str(z-deltaZ) + " and " + str(z+deltaZ) + " AND " \
                "t between " + str(t-deltaT) + " and " + str(t+deltaT) + " " \
          "group by (x, y, t)"

    print(sql)
    rows = DBUtil.querySQL(sql)

    if rows == None:
        return -1
    else:
        return rows, columns


'''
#  从数据库中选择合适的数据
#  z、t 值
#  deltaZ、deltaT 上下浮动区间
#  以[[x, y, z, t, temp]]输出
'''
def getTemperatureByZTGT(z, deltaZ, t, deltaT):
    columns = ['x', 'y', 'z', 't', 'temp']
    sql = "select avg(x), avg(y), avg(z), t, avg(temp) " \
          "from paper " \
          "where z between " + str(z-deltaZ) + " and " + str(z+deltaZ) + " AND " \
                "t between " + str(t-deltaT) + " and " + str(t+deltaT) + " " \
          "group by (t)"

    print(sql)
    rows = DBUtil.querySQL(sql)

    if rows == None:
        return -1
    else:
        return rows, columns


'''
#  从数据库中选择合适的数据
#  x、y、z、t 值
#  deltaZ、deltaT 上下浮动区间
#  以[[x, y, z, t, temp]]输出
'''
def getTemperatureByXYZT(x, deltaX, y, deltaY, z, deltaZ, t, deltaT):
    columns = ['x', 'y', 'z', 't', 'temp']
    sql = "select x, y, avg(z), t, avg(temp) " \
          "from paper " \
          "where x between " + str(x-deltaX) + " and " + str(x+deltaX) + " AND " \
                "y between " + str(y-deltaY) + " and " + str(y+deltaY) + " AND " \
                "z between " + str(z-deltaZ) + " and " + str(z+deltaZ) + " AND " \
                "t between " + str(t-deltaT) + " and " + str(t+deltaT) + " " \
          "group by (x, y, t)"

    print(sql)
    rows = DBUtil.querySQL(sql)

    if rows == None:
        return -1
    else:
        return rows, columns


def getTemperatureOnlyByXYZT(x, deltaX, y, deltaY, z, deltaZ, t, deltaT):
    columns = ['x', 'y', 'z', 't', 'temp']
    sql = "select x, y, z, t, temp " \
          "from paper " \
          "where x between " + str(x-deltaX) + " and " + str(x+deltaX) + " AND " \
                "y between " + str(y-deltaY) + " and " + str(y+deltaY) + " AND " \
                "z between " + str(z-deltaZ) + " and " + str(z+deltaZ) + " AND " \
                "t between " + str(t-deltaT) + " and " + str(t+deltaT) + " "

    print(sql)
    rows = DBUtil.querySQL(sql)

    if rows == None:
        return -1
    else:
        return rows, columns

def getTemperatureOnlyByXYZGT(x, deltaX, y, deltaY, z, deltaZ, t, deltaT):
    columns = ['x', 'y', 'z', 't', 'temp']
    sql = "select avg(x), avg(y), avg(z), t, avg(temp) " \
          "from paper " \
          "where x between " + str(x-deltaX) + " and " + str(x+deltaX) + " AND " \
                "y between " + str(y-deltaY) + " and " + str(y+deltaY) + " AND " \
                "z between " + str(z-deltaZ) + " and " + str(z+deltaZ) + " AND " \
                "t between " + str(t-deltaT) + " and " + str(t+deltaT) + " " \
          "group by (t)"

    print(sql)
    rows = DBUtil.querySQL(sql)

    if rows == None:
        return -1
    else:
        return rows, columns


'''
#  将数据转为csv文件
#  outputPath 输出csv文件路径
#  data  任意格式 [[1,2,3], ]
#  columns 与data格式一致['x', 'y', 'z']
'''
def saveAsCsv(outputPath, data, columns):
    print("saveAsCsv: ", outputPath)
    data = np.array(data)

    df = pd.DataFrame(columns=columns, data=data)
    df.to_csv(outputPath)
    return 0


'''
#  打开程序生成的csv，去掉第一列并输出为列表
#  csvFilePath csv路径
#  return data, title
'''
def readAsCsv(csvFilePath):
    data = pd.read_csv(csvFilePath, header=None).values.tolist()
    data = [row[1:] for row in data]

    columns = data[0]
    data = data[1:]

    return data, columns


'''
# 将列表转为shapefile
# outputPath 输出Shapefile文件路径
# data  任意格式 [[1,2,3], ]
# columns 与data格式一致['x', 'y', 'z']
'''
def saveAsShp(outputPath, data, columns, layerName='saveAsShp'):
    print("saveAsShp: ", outputPath)
    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)

    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.CreateDataSource(outputPath)
    layer = dataSource.CreateLayer(layerName, srs, ogr.wkbPoint)

    # Define Field
    columnsN = len(columns)
    for i in range(columnsN):
        field_name = ogr.FieldDefn(columns[i], ogr.OFTReal)
        # field_name.SetWidth(14)
        layer.CreateField(field_name)

    # data to feature
    for row in data:
        feature = ogr.Feature(layer.GetLayerDefn())
        for i in range(columnsN):
            feature.SetField(columns[i], str(row[i]))

        wkt = 'Point(' + str(row[0]) + ' ' + str(row[1]) + ')' # 特定的情况,点,01为xy
        point = ogr.CreateGeometryFromWkt(wkt)
        feature.SetGeometry(point)
        layer.CreateFeature(feature)

    dataSource = None
    return 0


'''
# 将数据导出为shapefile
# outputName FileName
# data[{x: y: v: }]
# layerName
def data2PointShp(outputName, data, layerName='data2shp'):
    driver = ogr.GetDriverByName('ESRI Shapefile')
    dataSource = driver.CreateDataSource(outputName)

    srs = osr.SpatialReference()
    srs.ImportFromEPSG(4326)
    layer = dataSource.CreateLayer(layerName, srs, ogr.wkbPoint)

    # Define Field
    field_name = ogr.FieldDefn("x", ogr.OFTReal)
    field_name.SetWidth(14)
    layer.CreateField(field_name)

    field_name = ogr.FieldDefn("y", ogr.OFTReal)
    field_name.SetWidth(14)
    layer.CreateField(field_name)

    field_name = ogr.FieldDefn("v", ogr.OFTReal)
    field_name.SetWidth(14)
    layer.CreateField(field_name)

    # data to feature
    for i in range(len(data)):
        tempFeat = data[i]
        feature = ogr.Feature(layer.GetLayerDefn())
        feature.SetField('x', float(tempFeat['x']))
        feature.SetField('y', float(tempFeat['y']))
        feature.SetField('v', float(tempFeat['v']))
        wkt = 'Point(' + str(tempFeat['x']) + ' ' + str(tempFeat['y']) + ')'
        point = ogr.CreateGeometryFromWkt(wkt)
        feature.SetGeometry(point)
        layer.CreateFeature(feature)

    dataSource = None
    return 0
'''


# ---------------------------------------------------------------------------------------
# IDW
'''
# 使用Grid
# outputPath 输出路径
# shapefilePath 点数据路径
'''
def IDW(outputPath, shapefilePath, option=None):
    print("IDW: ", outputPath)
    option = gdal.GridOptions(format='GTiff',
                          # width=2500,height=2500,
                          algorithm='invdist:power=3.6:smoothing=1:'
                                    'radius1=0.0:radius2=0.0:angle=0.0:'
                                    'max_points=15:min_points=0:nodata=0.0',
                          # layers=['myfirst'],
                          zfield='v'
                              )

    out = gdal.Grid(outputPath, shapefilePath, options=option)

    return 0



