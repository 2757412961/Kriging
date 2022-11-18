'''
    Name: Const Variable
    Creation: 2020-03-09
'''

# Const Variable
# 原始文件和csv导出文件路径
geoDirectory = "C:\\Users\\Z\\Desktop\\argo\\geo\\pacific_ocean"
csvDirectory = "F:\\EnglishPath\\1MyPaper\\4ETCData\\pacific_ocean"

# 输出文件夹
outputDirectory = "F:\\EnglishPath\\1MyPaper\\3ArcGIS"

# 第一天的juld
firstDay = 25202

# 研究范围
lon, dlon = 145, 35
lat, dlat = 0, 60

# depth = 250
# juldday = 25440  # 2019-08-27
# day = juldday - firstDay + 1

# ---------------------------------------------------------------------------------------
# Boundary
def getBoundary():
    return 110, 181, -60, 60

