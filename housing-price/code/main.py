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
from scipy import stats
from scipy.stats import  norm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

data_train = pd.read_csv("../data/train.csv")

#print(data_train)
print(data_train['SalePrice'].describe())
sns.distplot(data_train['SalePrice'])
plt.show()

