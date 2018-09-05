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
from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

data_train = pd.read_csv("../data/train.csv")

print(data_train,"\n")
print("SalePrice-describe:",data_train['SalePrice'].describe(),"\n")
sns.distplot(data_train['SalePrice'])
#plt.show()


print("Skewness: %f" % data_train['SalePrice'].skew())
print("Kurtosis: %f" % data_train['SalePrice'].kurt())
print("\n")

var = 'CentralAir'
data = pd.concat([data_train['SalePrice'],data_train[var]], axis=1)
fig = sns.boxplot(x = var, y = "SalePrice",data=data)
fig.axis(ymin = 0, ymax = 800000);
#plt.show()


var = 'OverallQual'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
#plt.show()


var = 'OverallQual'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
fig = sns.boxplot(x=var, y="SalePrice", data=data)
fig.axis(ymin=0, ymax=800000);
#plt.show()


# YearBuilt  scatter
var = 'YearBuilt'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
data.plot.scatter(x=var, y="SalePrice", ylim=(0, 800000))
#plt.show()


corrmat = data_train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.show()




