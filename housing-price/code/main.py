# -*- coding: utf-8 -*-
'''
Created on 2018年9月4日

@author: aeoogo
'''

# 导入需要的模块
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import warnings

from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler

def clear():
    os.system('cls')

#用来正常显示中文标签
plt.rcParams['font.sans-serif']=['SimHei'] 

#用来正常显示负号
plt.rcParams['axes.unicode_minus']=False 

warnings.filterwarnings('ignore')

#加载训练数据
data_train = pd.read_csv("../data/train.csv")

#预览数据
print(data_train,"\n")
input("press any key to continue......\n")
clear()

#房屋价格描述
print("SalePrice describe:",data_train['SalePrice'].describe(),"\n")
input("press any key to continue......\n")
clear()

#画出房屋价格分布密度
sns.distplot(data_train['SalePrice'])
plt.title(u'房屋价格分布密度')
plt.show()

# 相对于正态分布的峰度和偏度
print("Skewness(峰度): %f" % data_train['SalePrice'].skew())
print("Kurtosis(偏度): %f" % data_train['SalePrice'].kurt())
input("press any key to continue......\n")
clear()

#房价对于是否具有中央空调的箱型图
var = 'CentralAir'
data = pd.concat([data_train['SalePrice'],data_train[var]], axis=1)
fig = sns.boxplot(x = var, y = "SalePrice",data=data)
fig.axis(ymin = 0, ymax = 800000);
plt.title(u"房价对于是否具有中央空调的箱型图")
plt.show()

#房价对于总体评价的箱型图
var = 'OverallQual'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000)
plt.title(u"房价对于总体评价的箱型图")
plt.show()


#房价对于建造年份的箱型图
# YearBuilt boxplot
var = 'YearBuilt'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
f, ax = plt.subplots(figsize=(26, 12))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.title(u"房价对于建造年份的箱型图")

#房价对于建造年份的散点图
# YearBuilt  scatter
var = 'YearBuilt'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice", ylim=(0, 800000))
plt.title(u"房价对于建造年份的散点图")
plt.show()


#房价对于地段的箱型图
# Neighborhood
var = 'Neighborhood'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
f, ax = plt.subplots(figsize=(26, 12))
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
plt.title(u"房价对于地段的箱型图")
plt.show()


var  = 'LotArea'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))


var  = 'GrLivArea'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

var  = 'TotalBsmtSF'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

var  = 'MiscVal'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))

var  = ['GarageArea', 'GarageCars']
for index in range(2):
    data = pd.concat([data_train['SalePrice'], data_train[var[index]]], axis=1)
    data.plot.scatter(x=var[index], y='SalePrice', ylim=(0, 800000))

#各个属性的关系矩阵
corrmat = data_train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.title(u"各个属性的关系矩阵")
plt.show()


增加修改
