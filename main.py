# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 15:49:55 2023

@author: Algo1
"""
import os
import math
import re
import numpy as np
import baostock as bs
import tushare as ts
# pro = ts.pro_api('0b9e264dbea49edadaf3df7995fd913ce3d1422799d9652448947f82')
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from chinese_calendar import is_workday
import sys
import time
import warnings
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, HuberRegressor, ARDRegression
from sklearn.svm import SVR
from sklearn import metrics
warnings.filterwarnings('ignore')
sys.path.append("D:/学习资料/finance/")
sys.path.append("D:/学习资料/finance/code/")
import config as cfg
import utils as utl



#### 更新每日K线数据 ####
# utl.update_history_k()  

#### 更新财务报表指标 ####
# utl.update_stock_index(quarterEnd='20233')
df_stocks1 = utl.read_stocks(years=[2020,2021,2022], quarters=[4])
df_stocks2 = utl.read_stocks(years=[2023], quarters=[3])
df_stocks2 = df_stocks2[df_stocks2.NP>0] #净利润为正
df_stocks = pd.concat([df_stocks1, df_stocks2], ignore_index=True)
df_stocks['label']=1
df_top = utl.select_topStocks(df_stocks, ['WS', 3, 1000, 0.2], weightIdx=0, alpha=[1,2,3,6]) 
df_k = utl.read_historyK(codes=df_top.code, dateStart=30, dateEnd='')
df_k['label']=1
df_q = utl.cal_quantiles(df_k, idxBase=range(-1,0))
df_q = df_top[cfg.ColBase+['WS']].merge(df_q, how='right', on=cfg.ColBase)
pred = utl.pred_chg(df_top.code)



    





    




        



