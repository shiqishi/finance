# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:43:35 2023

@author: qishi
"""
import os
import sys
import numpy as np
import baostock as bs
import pandas as pd
import datetime
import time
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
sys.path.append("D:/学习资料/finance/code/")
import config as cfg

# 获取指定日期所有股票的日K线数据
def query_all_history_k(date, fields = cfg.FieldD , frequency = 'd', adjustflag = '2'):
    df_stock = bs.query_all_stock(date).get_data()
    if df_stock.empty:
        return  pd.DataFrame()
    ls_code = list(df_stock.code)
    ls_df = []
    t1 = time.time()
    for code in ls_code:
        df_k = bs.query_history_k_data_plus(code, fields, date, date, frequency, adjustflag).get_data()
        ls_df.append(df_k)
        progress_bar(len(ls_df), len(ls_code), '%s %04d'%(code, len(ls_df)))
    df_k = pd.concat(ls_df)
    df_k.set_index('code', inplace=True)
    t2 = time.time()
    progress_bar(len(ls_df), len(ls_code), '%s,%04d,%ds\n'%(date, len(ls_df), t2-t1), barLen=15)
    return df_k

# 更新每日K线数据
def update_history_k(deltaDay=-1):
    if (deltaDay<0 and datetime.datetime.now().hour>19):
        deltaDay = 0
    elif (deltaDay<0 and datetime.datetime.now().hour<19):
        deltaDay = 1
    dateEnd = (datetime.date.today() - datetime.timedelta(days=deltaDay)).strftime('%Y%m%d')
    bs.login()
    while True:
        files_historyK = sorted(os.listdir(cfg.PathHistoryK))
        if len(files_historyK)==0: #第一次运行初始化
            df_k = query_all_history_k('2018-01-02' )
            df_k.to_pickle(cfg.PathHistoryK+'20180102.pkl')
            files_historyK = sorted(os.listdir(cfg.PathHistoryK))
        dateLast = files_historyK[-1][:-4]
        if (dateLast>=dateEnd):
            print('history_k update finished !')
            break
        dateThis = (datetime.datetime.strptime(dateLast, "%Y%m%d")+datetime.timedelta(days=1)).strftime('%Y%m%d')
        df_k = query_all_history_k(datetime.datetime.strptime(dateThis, "%Y%m%d").strftime('%Y-%m-%d'))
        df_k.to_pickle(cfg.PathHistoryK+'%s.pkl'%dateThis) 
    bs.logout()

# 获取股票的所有财务报表指标
def query_all_stock_index(code, year, quarter):
    profit = bs.query_profit_data(code, year, quarter) #盈利能力
    profit = pd.DataFrame(profit.data, columns=profit.fields)
    if (profit.empty):
        return pd.DataFrame()
    operation = bs.query_operation_data(code, year, quarter) #营运能力
    operation = pd.DataFrame(operation.data, columns=operation.fields)
    growth = bs.query_growth_data(code, year, quarter) #成长能力 
    growth = pd.DataFrame(growth.data, columns=growth.fields)
    balance = bs.query_balance_data(code, year, quarter) #偿债能力
    balance = pd.DataFrame(balance.data, columns=balance.fields)   
    cash = bs.query_cash_flow_data(code, year, quarter)#现金流量
    cash = pd.DataFrame(cash.data, columns=cash.fields)   
    dupont = bs.query_dupont_data(code, year, quarter) #杜邦指数
    dupont = pd.DataFrame(dupont.data, columns=dupont.fields) 
    df_index = pd.concat([profit, operation, growth, balance, cash, dupont], axis=1)    
    if (not df_index.empty):
        df_index.insert(loc=0, column='code', value=code, allow_duplicates=True)
        df_index.insert(loc=0, column='quarter', value=quarter)
        df_index.insert(loc=0, column='year', value=year)
        df_index = df_index.T[~df_index.T.index.duplicated()].T.set_index('code') # 去掉重复的列
    return df_index

# 更新所有股票的财务报表指标
def update_stock_index(quarterEnd='', quarterStart='20124', ls_code=[]):
    # 获得默认的更新截止季度
    if len(quarterEnd)!=5:
        year = int(datetime.date.today().strftime('%Y'))
        quarter = (int(datetime.date.today().strftime('%m'))+2)//3-1
        if quarter<1:
            year = year-1
            quarter = quarter+4  
        quarterEnd = '%s%s'%(year, quarter) 
    # 获得所要更新的code列表
    if len(ls_code)==0:
        files_stocks = os.listdir(cfg.PathStocks)
        ls_code = [stock[:9] for stock in files_stocks]
    bs.login()
    quarterCount = 0
    while(True):
        quarterCount += 1
        # 获得已更新的信息，避免重复更新
        files_stocks = os.listdir(cfg.PathStocks)
        codes = [stock[:9] for stock in files_stocks]
        codeCount = 0
        updateCount = 0
        t1 = time.time()
        for code in ls_code:
            codeCount += 1
            if code in codes:
                file_stock = files_stocks[codes.index(code)]
            else:
                file_stock =  '%s[%s].csv'%(code, quarterStart) #以2013第1季度为更新开始季度，
            year = int(file_stock[10:14])
            quarter = int(file_stock[14:15])
            while('%s%s'%(year, quarter)<quarterEnd): 
                quarter += 1
                if quarter>4:
                    year = year+1
                    quarter = quarter-4 
                progress_bar(codeCount, len(ls_code), '%s %02d%02d'%(code, year-2000, quarter))
                file_new = '%s[%s%s].csv'%(code, year, quarter)
                df_index = query_all_stock_index(code, year, quarter)
                if (df_index.shape[0]>0):
                    df_index.to_csv(cfg.PathStocks+file_stock, encoding="gbk", index=True, header=code not in codes, mode='a') 
                    os.renames(cfg.PathStocks+file_stock, cfg.PathStocks+file_new)
                    updateCount += 1
                    break
        t2 = time.time()
        progress_bar(codeCount, len(ls_code), '%d/%d, %ds\n'%(updateCount, codeCount, t2-t1))
        if updateCount==0:
            print('stocks update finished !')
            break
    bs.logout()
    
def read_basic():
    if os.path.exists(cfg.FileBasic):
        df_basic = pd.read_csv(cfg.FileBasic, encoding='gbk')
    else:
        bs.login()
        df_basic = bs.query_stock_industry().get_data() # 获取申万分类信息(每周一更)
        bs.logout()
        df_amount = df_basic.value_counts('industry', ascending=False).to_frame('amount').reset_index()
        df_amount.insert(loc=0, column='label', value=['%02d'%i for i in range(1,len(df_amount)+1)])
        df_basic = df_basic.merge(df_amount, how='left', on='industry')        
        df_basic.to_csv(cfg.FileBasic, encoding="gbk", index=False, mode='w')
    print('have read basic![%03d]'%len(df_basic))
    return df_basic

def read_stocks(codes=[], years=[2020,2021,2022], quarters=[4]):
    if os.path.exists(cfg.FileStock):
        df_stocks = pd.read_pickle(cfg.FileStock)
    else:
        files = os.listdir(cfg.PathStocks)
        ls_df = []
        finish_number = 0
        for file in files:                   
            ls_df.append(pd.read_csv(cfg.PathStocks+file, encoding='gbk'))  
            progress_bar(finish_number, len(files), msg=file[:9])
            finish_number += 1
        df_stocks = pd.concat(ls_df,ignore_index=True) 
        df_basic = read_basic()
        df_stocks = df_stocks.merge(df_basic[cfg.ColBase],how='left',on='code') 
        df_stocks.to_pickle(cfg.FileStock)
    if len(codes)>0:
        df_stocks = df_stocks[df_stocks.code.isin(codes)]
    if len(years)>0:
        df_stocks = df_stocks[df_stocks.year.isin(years)]
    if len(quarters)>0:
        df_stocks = df_stocks[df_stocks.quarter.isin(quarters)]
    colOld = cfg.ColBase + cfg.ColDate + [col[0] for col in cfg.ColWs.keys()]
    colNew = cfg.ColBase + cfg.ColDate + [col[1] for col in cfg.ColWs.keys()]
    df_stocks = df_stocks[colOld]
    df_stocks.columns = colNew
    df_stocks = df_stocks.round(3)
    df_stocks['NP'] = df_stocks['NP']/10**6
    df_stocks['NP'] = df_stocks['NP'].apply(lambda x : np.round(x,2) if abs(x)<100 else np.round(x))                                                      
    progress_bar(1, 1, msg='read stocks![%s]\n'%len(df_stocks), barLen=5) 
    return df_stocks
    
def transfer_to_score(df_stocks=pd.DataFrame()):
    if df_stocks.empty:
       df_stocks = read_stocks() 
    ls_df = []
    Cols = [col[1] for col in cfg.ColWs.keys()] 
    quantiles = [0.00,0.025,0.05,0.10,0.20,0.40,0.60,0.80,0.90,0.95,0.975,1.0]
    gb_stocks = df_stocks.groupby(by=['year','quarter','label'])
    for info, df in gb_stocks:
        for col in Cols:
            bins, labels = np.unique(df[col].dropna().quantile(quantiles, interpolation='linear'), return_index=True)
            labels = (labels[0:-1] + labels[1:])/2-0.5
            df[col] = pd.cut(df[col], bins=bins, labels=labels, right=True, include_lowest=True)
        progress_bar(len(ls_df), len(gb_stocks), msg=str(info)) 
        ls_df.append(df)
    df_scores = pd.concat(ls_df, ignore_index=True)
    df_scores[Cols] = df_scores[Cols].astype(float)
    progress_bar(1, 1, msg='transfered to scores!\n') 
    # df_describe = df_scores.describe()
    return df_scores

def cal_weightScore(df_scores, weightIdx=0, alpha=1.4):
    ls_df = []
    Cols = [col[1] for col in cfg.ColWs.keys()]     
    Weights = [val[weightIdx] for val in cfg.ColWs.values()]  
    df_scores.sort_values(by=['code','statDate'],inplace=True)
    gb_scores = df_scores.groupby(by= 'code')
    for code, df in gb_scores: 
        if isinstance(alpha, int) or isinstance(alpha, float):
            alphas = alpha ** np.arange(1, len(df)+1)            
        elif isinstance(alpha, list) and len(alpha)==len(df):
            alphas = alpha
        else:
            continue
        alphas = np.expand_dims(alphas,1).repeat(len(Cols),1)    
        alphas[df[Cols].isna().to_numpy()]=0 #缺失值的权重为0
        alphas = alphas/alphas.sum(axis=0)
        df_sum= pd.DataFrame(df[cfg.ColBase][-1:], columns=cfg.ColBase)
        vals = np.array((df[Cols]*alphas).sum(numeric_only=True,min_count=1))
        weights = np.where(np.isnan(vals), 0, Weights) #缺失值的权重为0
        df_sum['WS'] = np.nansum(vals*weights/sum(weights))
        df_sum[Cols] =  vals 
        progress_bar(len(ls_df), len(gb_scores), msg=code)
        ls_df.append(df_sum)
    df_ws = pd.concat(ls_df, ignore_index=True)
    progress_bar(1, 1, msg='calculated weighted sum!\n')    
    return df_ws

def select_topStocks(df_stocks=pd.DataFrame(), topSelect=['WS', 3, 20, 0.05], weightIdx=0, alpha=1.4, filter68=True):
    if df_stocks.empty:
        df_stocks = read_stocks() 
    fileTop = cfg.PathTemp + '%s%s[%s,%s,%s].xlsx'%(str(topSelect), len(df_stocks), weightIdx, alpha, filter68)
    if os.path.exists(fileTop):
        df_top = pd.read_excel(fileTop)
        print('have read topStocks![%03d]'%len(df_top))
    else:   
        df_scores = transfer_to_score(df_stocks) 
        df_ws = cal_weightScore(df_scores, weightIdx, alpha) 
        ls_df = []
        for label, df in df_ws.groupby(by='label'):
            df.sort_values(by=topSelect[0],ascending=False, inplace=True, ignore_index=True)
            df.insert(len(cfg.ColBase),'total',len(df))
            df = df[0:max(topSelect[1], min(topSelect[2], round(len(df)*topSelect[3])))] #top5% 且 前3到前10名
            if filter68:
                df = df[df.code.str[0:5]!='sh.68'] # 过滤掉科技股
            ls_df.append(df)
        df_top = pd.concat(ls_df, ignore_index=True)
        df_top = df_top.round(2)
        df_top.to_excel(fileTop,float_format='%.2f', index=False)
        print('have select top stocks![%03d]'%len(df_top))
    return df_top

# 获得目标股票的历史日K线数据
def read_historyK(codes=[], dateStart=365, dateEnd=''):
    dates = [file[:-4] for file in sorted(os.listdir(cfg.PathHistoryK))]
    df_basic = read_basic()
    if type(dateEnd) is int:
        dateEnd = (datetime.date.today() - datetime.timedelta(days=dateEnd)).strftime('%Y%m%d')
    if len(dateEnd)!=8 or dateEnd>dates[-1]:
        dateEnd = dates[-1]   
    if type(dateStart) is int:
       dateStart = (datetime.datetime.strptime(dateEnd, "%Y%m%d") - datetime.timedelta(days=dateStart)).strftime('%Y%m%d') 
    if len(dateStart)!=8:
        dateStart = (datetime.datetime.strptime(dateEnd, "%Y%m%d") - datetime.timedelta(days=365)).strftime('%Y%m%d')
    if len(codes)==0:
        codes = list(df_basic.code)
    fileHistoryK = cfg.PathTemp + '%03d[%s][%s].pkl'%(len(codes), dateStart, dateEnd)
    if os.path.exists(fileHistoryK):
        df_k = pd.read_pickle(fileHistoryK)
        print('have read historyK![%03d]'%len(df_k))
    else:  
        ls_df = []
        dates = [date for date in dates if (date>=dateStart) & (date<=dateEnd)]
        finish_number = 0
        for date in dates:
            df_k = pd.read_pickle(cfg.PathHistoryK+'%s.pkl'%date)
            finish_number += 1
            progress_bar(finish_number, len(dates), msg=date, barLen = 25) 
            if not(df_k.empty):
                df_k = df_k[df_k.index.isin(codes)]
                df_k.reset_index(inplace=True)
                df_k.replace(to_replace=r'^\s*$',value=np.nan,regex=True,inplace=True) #填充空字符串
                df_k = df_k.astype({'open':float, 'high':float,'low':float,'close':float, 'preclose':float, 
                                    'volume':float, 'amount':float,'adjustflag':int,'turn':float,'tradestatus':int,
                                    'pctChg':float,'peTTM':float,'pbMRQ':float,'psTTM':float,'pcfNcfTTM':float, 'isST':int})        
                ls_df.append(df_k)
        df_k = pd.concat(ls_df, ignore_index=True)
        df_k = df_k[df_k.code.isin(df_k[df_k.date == max(set(df_k.date))].code)] #只保留最后一天存在的股票
        colK = cfg.ColBase + list(df_k.columns[1:])
        df_k = df_k.merge(df_basic[cfg.ColBase],how='left',on='code')
        df_k = df_k[colK]
        df_k.sort_values(by=['date', 'code'], inplace=True, ignore_index=True)
        codes, indexs = np.unique(df_k.code, return_index=True) #删除第1天的数据
        df_k.drop(indexs,inplace=True)
        df_k.reset_index(drop=True, inplace=True)
        # indexs = df_k.code.str[3:5].isin(['30', '68']) # 科技股和创业股的日涨幅除以2
        # df_k.loc[indexs, 'pctChg'] = 0.5*df_k[indexs].pctChg        
        df_k.to_pickle(fileHistoryK)
        progress_bar(1, 1, msg='selected historyK[%d]\n'%len(df_k), barLen = 25) 
    return df_k

def cal_quantiles(df_k=pd.DataFrame(), idxBase=range(-1,0)):
    quantiles = [0.95,0.80,0.50, 0.20, 0.05]
    if df_k.empty:
        df_k = read_historyK()
    gb_k = df_k.groupby(by='label')
    ls_df = []
    finish_number = 0
    for label, df_label in gb_k:
        df_mean = df_label.groupby(by='date').mean()
        df_base = df_mean.iloc[idxBase].mean()
        qs = np.array(df_mean.close.quantile(quantiles, interpolation='linear'))
        qr = (df_base.close/qs-1)*100
        df = df_label[cfg.ColBase][:1]
        df[['code', 'code_name']] = 'mean'
        df['GM'] = np.round(np.mean(qr),2)
        df[quantiles] = np.round(qr,2)   
        ls_df.append(df)
        for code, df_code in df_label.groupby(by='code'):
            df_base = df_code.iloc[idxBase].mean()
            qs = np.array(df_code.close.quantile(quantiles, interpolation='linear'))
            qr = (df_base.close/qs-1)*100
            df = df_code[cfg.ColBase][:1]
            df['GM'] = np.round(np.mean(qr),2)
            df[quantiles] =  np.round(qr,2)
            ls_df.append(df)
        progress_bar(finish_number, len(gb_k), msg=str(label), barLen = 25) 
        finish_number += 1
    df_q = pd.concat(ls_df, ignore_index=True)
    progress_bar(1, 1, msg='calulated quantiles![%d]\n'%len(df_q), barLen=25)
    return df_q

def chg_data(df_k):
    ls_df = []
    gb_k = df_k.groupby('code')
    for code, df_code in gb_k:
        ls = df_code.close.values
        if len(ls)-5<=250:
            continue
        df = df_code[cfg.ColBase+['date']][250:]
        progress_bar(len(ls_df), len(gb_k), msg=code, barLen=25)
        for Idx in range(250,len(ls)-1):
            qy = ls[Idx] / np.quantile(ls[Idx-250:Idx], [0.05,0.5,0.95])-1
            qm = ls[Idx] / np.quantile(ls[Idx-25:Idx], [0.05,0.5,0.95])-1
            qd = np.array([ls[Idx-2]/ls[Idx-3], ls[Idx-1]/ls[Idx-2], ls[Idx]/ls[Idx-1]])-1
            qn = ls[Idx+1]/ls[Idx]-1
            if Idx==250:
                qs = np.r_[qy, qm, qd, qn]
            qs = np.row_stack((qs, np.r_[qy, qm, qd, qn]))
        df[['y5','y50','y95','m5','m50','m95','d2','d1','d0','n1']]= qs
        ls_df.append(df) 
    if len(ls_df)>0:
        df_chg = pd.concat(ls_df,ignore_index=True)
        df_chg.sort_values(by='date', inplace=True, ignore_index=True)
    else:
        df_chg = pd.DataFrame()
    progress_bar(1, 1, msg='done![%04d]\n'%len(df_chg), barLen = 25)
    return df_chg

def pred_chg(codes, trainNum=40, regressor=KNeighborsRegressor(n_neighbors=10, leaf_size=10)):
    # 获取Khistory数据
    dates = [file[:-4] for file in sorted(os.listdir(cfg.PathHistoryK), reverse=True)]
    df_basic = read_basic()
    ls_df = []
    for date in dates:
        progress_bar(len(ls_df), trainNum+251, msg=date, barLen=25)
        df_k = pd.read_pickle(cfg.PathHistoryK+'%s.pkl'%date)
        if df_k.empty:
            continue
        df_k = df_k[df_k.index.isin(codes)]
        df_k.reset_index(inplace=True)
        df_k.replace(to_replace=r'^\s*$',value=np.nan,regex=True,inplace=True) #填充空字符串
        df_k = df_k.astype({'open':float, 'high':float,'low':float,'close':float, 'preclose':float, 
                            'volume':float, 'amount':float,'adjustflag':int,'turn':float,'tradestatus':int,
                            'pctChg':float,'peTTM':float,'pbMRQ':float,'psTTM':float,'pcfNcfTTM':float, 'isST':int})        
        ls_df.append(df_k)
        if len(ls_df)>trainNum+251:
            progress_bar(1, 1, msg='get Khistory data!\n', barLen=25)
            break
    df_k = pd.concat(ls_df, ignore_index=True)
    df_k = df_k[df_k.code.isin(df_k[df_k.date == max(set(df_k.date))].code)] #只保留最后一天存在的股票
    colK = cfg.ColBase + list(df_k.columns[1:])
    df_k = df_k.merge(df_basic[cfg.ColBase],how='left',on='code')
    df_k = df_k[colK]
    df_k.sort_values(by=['date', 'code'], inplace=True, ignore_index=True)
    codes, indexs = np.unique(df_k.code, return_index=True) #删除第1天的数据
    df_k.drop(indexs,inplace=True)
    df_k.reset_index(drop=True, inplace=True)
    # indexs = df_k.code.str[3:5].isin(['30', '68']) # 科技股和创业股的日涨幅除以2 
    # 获取chg数据
    gb_k = df_k.groupby('code')
    x_train = np.empty((0,9))
    y_train = []
    x_pred = np.empty((0,9))
    code_pred = []
    for code, df_code in gb_k:
        progress_bar(len(x_pred), len(gb_k), msg=code, barLen=25)
        ls = df_code.close.values
        if len(ls)<=252:
            continue
        for Idx in range(250,len(ls)-1):
            qy = ls[Idx] / np.quantile(ls[Idx-250:Idx], [0.05,0.5,0.95])-1
            qm = ls[Idx] / np.quantile(ls[Idx-25:Idx], [0.05,0.5,0.95])-1
            qd = np.array([ls[Idx-2]/ls[Idx-3], ls[Idx-1]/ls[Idx-2], ls[Idx]/ls[Idx-1]])-1
            x_train = np.row_stack((x_train, np.r_[qy, qm, qd]))
            y_train.append(ls[Idx+1]/ls[Idx]-1)
        qy = ls[-1] / np.quantile(ls[-250:], [0.05,0.5,0.95])-1
        qm = ls[-1] / np.quantile(ls[-25:], [0.05,0.5,0.95])-1
        qd = np.array([ls[-3]/ls[-4], ls[-2]/ls[-3], ls[-1]/ls[-2]])-1    
        x_pred = np.row_stack((x_pred, np.r_[qy, qm, qd]))
        code_pred.append(code)
    progress_bar(1, 1, msg='get chg data!\n', barLen=25)
    regressor.fit(x_train, y_train)
    y_pred = np.round(regressor.predict(x_pred)*100, 2)
    pred= dict(zip(code_pred, y_pred))    
    pred = sorted(pred.items(),key = lambda x:x[1],reverse = True)
    return pred

def experiment_trainModel(df_chg, regressor=KNeighborsRegressor(n_neighbors=10, leaf_size=10), experNum=100, trainNum=40):
    dates = sorted(list(set(df_chg.date)))
    results = {'base':[], 'pred':[]}
    for Idx in range(len(dates)-experNum,len(dates)):
        progress_bar(Idx+experNum-len(dates), len(dates), msg='', barLen=25)
        b_train = df_chg.date.isin(dates[Idx-trainNum:Idx-1])
        b_test = df_chg.date==dates[Idx]
        x_train = df_chg[b_train][['y5','y50','y95','m5','m50','m95','d2','d1','d0']].to_numpy()
        y_train = df_chg[b_train]['n1'].to_numpy()
        x_test = df_chg[b_test][['y5','y50','y95','m5','m50','m95','d2','d1','d0']].to_numpy()
        y_test = df_chg[b_test]['n1'].to_numpy()    
        regressor.fit(x_train,y_train)    
        y_pred = regressor.predict(x_test)
        results['base'].append(np.round(np.mean(y_test)*100,2))
        results['pred'].append(np.round(np.mean(y_test[np.argsort(-y_pred)[:5]])*100,2))
    df_lift = pd.DataFrame.from_dict(results)
    df_lift['lift'] = df_lift['pred']-df_lift['base']
    progress_bar(1, 1, msg='%.2f\n'%df_lift.mean(axis=0)['lift'], barLen=25)
    df_lift.loc['mean',:] = df_lift.mean(axis=0)
    return df_lift

def plot_k(df_k, dirName='trendChart_k', cols = ['close','peTTM','pbMRQ','psTTM','pcfNcfTTM'], linewidth=8):
    finish_number = 0
    gb_k = df_k.groupby(by='code')
    pathDir = os.path.join(cfg.PathTemp, dirName)
    if not os.path.exists(pathDir):
        os.makedirs(pathDir)
    for code, df in gb_k:
        df.reset_index(drop=True, inplace=True)
        df.plot(kind='line', linewidth=linewidth, sharex=True, y=cols, subplots=True, layout=(len(cols),1), figsize=(20,20))
        plt.savefig(os.path.join(pathDir, '%s.png'%code))
        finish_number += 1
        progress_bar(finish_number, len(gb_k), msg=code, barLen = 25)
    progress_bar(finish_number, len(gb_k), msg='done![%04d]\n'%len(gb_k), barLen = 25) 
    
def progress_bar(finish_number, tasks_number, msg='', barLen = 20):
    percentage = finish_number / tasks_number * 100
    finishLen = round(finish_number / tasks_number * barLen)
    print("\r{:04.1f}% [{}{}] {:10s}".format(percentage, "▓"*finishLen, "-"*(barLen-finishLen), msg), end="")  
    
"""
def query_all_history_k(date, fields = FieldD , frequency = 'd', adjustflag = '2'):
    lg = bs.login()
    df_stock = bs.query_all_stock(date).get_data()
    if df_stock.empty:
        return  pd.DataFrame()
    ls_df = []
    sem = Semaphore(100)
    def myThread(code, fields, date, frequency, adjustflag):       
        with sem:
            df_k = bs.query_history_k_data_plus(code, fields, date, date, frequency, adjustflag).get_data()
            ls_df.append(df_k)
            print('%s, %s//%s'%(code, len(ls_df),len(df_stock)))
    ls_thread = []
    for code in list(df_stock.code)[:100]:
        thread = Thread(target = myThread, args=(code, fields, date, frequency, adjustflag))
        thread.start()
        ls_thread.append(thread)
    for t in ls_thread:
        t.join()
    bs.logout()
    df_k = pd.concat(ls_df)
    df_k.set_index('code', inplace=True)
    df_k.sort_index(inplace=True)
    return ls_df  
"""  


