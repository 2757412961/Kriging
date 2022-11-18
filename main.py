from Utils import *
from ConstVariable import *
import STKrig
import STNNKrig
import STNNKrig2
import STNNKrig3
import LSTM
import GeoBiLSTM


def stkrig(depth, juldday, day, name):
    now = getNow()
    # now = name
    ### step 1
    # ExtractAllNc2Csv(geoDirectory, csvDirectory)
    # importAllcsv2pgDB(csvDirectory)
    ### step 2
    data, columns = getTemperatureByXYZT(lon, dlon, lat, dlat, depth, 6, juldday, 15)
    saveAsCsv(outputDirectory + '\\STKrig_' + now + "_RAW.csv", data, columns)
    saveAsShp(outputDirectory + '\\STKrig_' + now + "_RAW.shp", data, columns)
    ### step 3
    outputData, outputColumns = STKrig.SpatioTemporalKriging(data, columns, depth, day)
    # saveAsCsv(outputDirectory + '\\STKrig_' + now + ".csv", outputData, outputColumns)
    saveAsShp(outputDirectory + '\\STKrig_' + now + ".shp", outputData, outputColumns)
    ### step 4
    IDW(outputDirectory + '\\STKrig_' + now + ".tif",
        outputDirectory + '\\STKrig_' + now + ".shp")


def stnnkrig(depth, juldday, day, name):
    now = getNow()
    # now = name
    ### step 1
    dataS, columns = getTemperatureByXYZT(lon, dlon, lat, dlat, depth, 6, juldday, 5)
    dataT, columns = getTemperatureByZTGT(depth, 6, juldday, 60)
    data, columns = getTemperatureByXYZT(lon, dlon, lat, dlat, depth, 6, juldday, 15)
    saveAsCsv(outputDirectory + '\\STNNKrig_' + now + "_RAW.csv", data, columns)
    saveAsShp(outputDirectory + '\\STNNKrig_' + now + "_RAW.shp", data, columns)
    ### step 2
    outputData, outputColumns = \
        STNNKrig.SpatioTemporalNeuralNetworkKriging(dataS, dataT, data, columns, depth, day)
    # saveAsCsv(outputDirectory + '\\STNNKrig_' + now + ".csv", outputData, outputColumns)
    saveAsShp(outputDirectory + '\\STNNKrig_' + now + ".shp", outputData, outputColumns)
    ### step 3
    IDW(outputDirectory + '\\STNNKrig_' + now + ".tif",
        outputDirectory + '\\STNNKrig_' + now + ".shp")

    return


def stnnkrig2(depth, juldday, day, name):
    now = getNow()
    ### step 1
    data, columns = getTemperatureByXYZT(lon, dlon, lat, dlat, depth, 6, juldday, 15)
    saveAsCsv(outputDirectory + '\\STNNKrig2_' + now + "_RAW.csv", data, columns)
    saveAsShp(outputDirectory + '\\STNNKrig2_' + now + "_RAW.shp", data, columns)
    ### step 2
    outputData, outputColumns = \
        STNNKrig2.SpatioTemporalNeuralNetworkKriging(data, columns, depth, day)
    # saveAsCsv(outputDirectory + '\\STNNKrig2_' + now + ".csv", outputData, outputColumns)
    saveAsShp(outputDirectory + '\\STNNKrig2_' + now + ".shp", outputData, outputColumns)
    ### step 3
    IDW(outputDirectory + '\\STNNKrig2_' + now + ".tif",
        outputDirectory + '\\STNNKrig2_' + now + ".shp")

    return


def stnnkrig3(depth, juldday, day, name):
    now = getNow()
    ### step 1
    data, columns = getTemperatureByXYZT(lon, dlon, lat, dlat, depth, 6, juldday, 15)
    saveAsCsv(outputDirectory + '\\STNNKrig3_' + now + "_RAW.csv", data, columns)
    saveAsShp(outputDirectory + '\\STNNKrig3_' + now + "_RAW.shp", data, columns)
    ### step 2
    outputData, outputColumns = \
        STNNKrig3.SpatioTemporalNeuralNetworkKriging(data, columns, depth, day)
    # saveAsCsv(outputDirectory + '\\STNNKrig3_' + now + ".csv", outputData, outputColumns)
    saveAsShp(outputDirectory + '\\STNNKrig3_' + now + ".shp", outputData, outputColumns)
    ### step 3
    IDW(outputDirectory + '\\STNNKrig3_' + now + ".tif",
        outputDirectory + '\\STNNKrig3_' + now + ".shp")

    return


def lstm(depth, juldday, day, name):
    now = getNow()
    ### step 1
    outputData, outputColum = LSTM.Grid(depth, day)
    # saveAsCsv(outputDirectory + '\\LSTM_' + now + ".csv", outputData, outputColum)
    saveAsShp(outputDirectory + '\\LSTM_' + now + ".shp", outputData, outputColum)
    ### step 2
    IDW(outputDirectory + '\\LSTM_' + now + ".tif",
        outputDirectory + '\\LSTM_' + now + ".shp")
    ### step 3
    # 最后用ArcGIS来处理


def gbilstm(depth, juldday, day, name):
    now = getNow()
    # now = name
    ### step 1
    data, columns = getTemperatureByXYZT(lon, dlon, lat, dlat, depth, 6, juldday, 15)
    saveAsCsv(outputDirectory + '\\GBiLSTM_' + now + "_RAW.csv", data, columns)
    saveAsShp(outputDirectory + '\\GBiLSTM_' + now + "_RAW.shp", data, columns)
    ### step 2
    outputData, outputColumns = \
        GeoBiLSTM.GeographyBiLSTM(data, columns, depth, day)
    # saveAsCsv(outputDirectory + '\\GBiLSTM_' + now + ".csv", outputData, outputColumns)
    saveAsShp(outputDirectory + '\\GBiLSTM_' + now + ".shp", outputData, outputColumns)
    ### step 3
    IDW(outputDirectory + '\\GBiLSTM_' + now + ".tif",
        outputDirectory + '\\GBiLSTM_' + now + ".shp")

    return


# main---------------------------------------------------------------------------
# stkrig(250, 25440, 25440 - firstDay + 1, "0name")
stnnkrig(250, 25440, 25440 - firstDay + 1, "0name")  # 效果一般
# stnnkrig2() # 效果不可以
# stnnkrig3() # 效果不行
# lstm()      # 效果不好
# gbilstm(250, 25440, 25440 - firstDay + 1, "0name") # 双向LSTM

# 10  --冬季 1月份
# 100 --春季 3月份
# 190 --夏季 6月份
# 280 --秋季 9月份
def main():
    # juldday = 25440  # 2019-08-27
    # firstDay = 25202

    # months = [25211, 25241, 25271, 25301,
    #           25331, 25361, 25391, 25421,
    #           25451, 25481, 25511, 25541]
    # months = [25481]
    # for juldday in months:
    #     depth = 250
    #     day = juldday - firstDay + 1
    #     stkrig(depth, juldday, day, "0day"+str(day))    # 效果可以
    #     stnnkrig(depth, juldday, day, "0day"+str(day)) # 效果一般
    #     gbilstm(depth, juldday, day, "0day"+str(day))   # 双向LSTM
    #     print("Now time: " + getNow())

    depths = [] # [600]
    for depth in depths:
        juldday = 25481
        day = juldday - firstDay + 1
        # stkrig(depth, juldday, day, "depth"+str(depth))    # 效果可以
        stnnkrig(depth, juldday, day, "depth"+str(depth)) # 效果一般
        # gbilstm(depth, juldday, day, "depth"+str(depth))   # 双向LSTM
        # print("Now time: " + getNow())


if __name__ == '__main__':
    main()

exit(0)
