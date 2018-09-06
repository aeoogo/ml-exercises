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
from pandas import Series, DataFrame

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
#print(data_train.describe())
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
#var = 'YearBuilt'
#data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
#f, ax = plt.subplots(figsize=(26, 12))
#fig = sns.boxplot(x=var, y="SalePrice", data=data)
#fig.axis(ymin=0, ymax=800000);
#plt.title(u"房价对于建造年份的箱型图")
#plt.show()

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



# 房价对于地表面积的散点图
var  = 'LotArea'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.title(u"房价对于地表面积的散点图")
plt.show()

#房价对于GrLivArea的散点图 
var  = 'GrLivArea'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.title(u"房价对于Grlivarea的散点图")
plt.show()


#房价与TotalBsmtSF之间的散点图
var  = 'TotalBsmtSF'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.title(u"房价与totalbsmtsf之间的散点图")
plt.show()

#房价与MiscVal之间的散点图
var  = 'MiscVal'
data = pd.concat([data_train['SalePrice'], data_train[var]], axis=1)
data.plot.scatter(x=var, y='SalePrice', ylim=(0, 800000))
plt.title(u"房价与Miscval之间的散点图")
plt.show()



var  = ['GarageArea', 'GarageCars']
for index in range(2):
    data = pd.concat([data_train['SalePrice'], data_train[var[index]]], axis=1)
    data.plot.scatter(x=var[index], y='SalePrice', ylim=(0, 800000))
    plt.title(u"房价与"+var[index]+u"之间的关系")
    plt.show()


#各个属性的关系矩阵
corrmat = data_train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.title(u"各个属性的关系矩阵")
plt.show()



#各个属性的关系矩阵（包含离散型数据）
from sklearn import preprocessing
f_names = ['CentralAir', 'Neighborhood']
for x in f_names:
    label = preprocessing.LabelEncoder()
    data_train[x] = label.fit_transform(data_train[x])
corrmat = data_train.corr()
f, ax = plt.subplots(figsize=(20, 9))
sns.heatmap(corrmat, vmax=0.8, square=True)
plt.title(u"各个属性的关系矩阵（包含离散性数据）")
plt.show()



#与价格相关性最大的10个特征
k  = 10 # 关系矩阵中将显示10个特征
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(data_train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True,square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.title("ten most relevant features")
plt.show()


#绘制选择的特征的关系点图
sns.set()
cols = ['SalePrice','OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
sns.pairplot(data_train[cols], size = 1.5)
plt.title(u"选择的特征的关系点图")
plt.show()





from sklearn import preprocessing
from sklearn import linear_model, svm, gaussian_process
from sklearn.ensemble import RandomForestRegressor
from sklearn.cross_validation import train_test_split
import numpy as np

# 获取数据
cols = ['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
x = data_train[cols].values
y = data_train['SalePrice'].values
x_scaled = preprocessing.StandardScaler().fit_transform(x)
y_scaled = preprocessing.StandardScaler().fit_transform(y.reshape(-1,1))
X_train,X_test, y_train, y_test = train_test_split(x_scaled, y_scaled, test_size=0.33, random_state=42)


clfs = {
        'svm':svm.SVR(), 
        'RandomForestRegressor':RandomForestRegressor(n_estimators=400),
        'BayesianRidge':linear_model.BayesianRidge()
       }
for clf in clfs:
    try:
        clfs[clf].fit(X_train, y_train)
        y_pred = clfs[clf].predict(X_test)
        print(clf + " cost:" + str(np.sum(abs(y_pred-y_test))/len(y_pred)) )
    except Exception as e:
        print(clf + " Error:")
        print(str(e))



#未归一化的预测结果
cols = ['OverallQual','GrLivArea', 'GarageCars','TotalBsmtSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
x = data_train[cols].values
y = data_train['SalePrice'].values
X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=42)

clf = RandomForestRegressor(n_estimators=400)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print(u"未归一化的验证集数据预测结果\n")
print(y_pred)
print(u"验证集数据实际价格\n")
print(y_test)



vali_result = pd.concat([DataFrame(y_pred,columns=['predict']),DataFrame(y_test,columns=['actual'])], axis=1)
vali_result.to_csv('../data/Validation-Prediction.csv', index=False)





print("平均绝对误差:",sum(abs(y_pred - y_test))/len(y_pred))
print("\n")





#验证测试集数据
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# 之前训练的模型
rfr = clf

data_test = pd.read_csv("../data/test.csv")
data_test[cols].isnull().sum()



data_test['GarageCars'].describe()
data_test['TotalBsmtSF'].describe()



cols2 = ['OverallQual','GrLivArea', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt']
cars = data_test['GarageCars'].fillna(1.766118)
bsmt = data_test['TotalBsmtSF'].fillna(1046.117970)
data_test_x = pd.concat( [data_test[cols2], cars, bsmt] ,axis=1)
data_test_x.isnull().sum()



x = data_test_x.values
y_te_pred = rfr.predict(x)
print(y_te_pred)

print(y_te_pred.shape)
print(x.shape)



prediction = pd.DataFrame(y_te_pred, columns=['SalePrice'])
result = pd.concat([ data_test['Id'], prediction,], axis=1)
# result = result.drop(resultlt.columns[0], 1)
result.columns




result.to_csv('../data/Predictions.csv', index=False)


就改那么一点

