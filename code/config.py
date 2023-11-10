# -*- coding: utf-8 -*-
"""
Created on Fri Sep 15 16:47:47 2023

@author: Algo1
"""
PathTemp = "D:/学习资料/finance/temp/"
PathData = "D:/学习资料/finance/data/"
PathHistoryK = PathData + "history_k/"  #每个日K线的存储路径
PathStocks =  PathData + "stocks/"  #每个证券财务指标的存储路径
FileBasic = PathData + "basic.csv"
FileStock = PathData+'stocks.pkl'
FieldD = "date,code,open,high,low,close,preclose,volume,amount,adjustflag,\
          turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM,isST"  #日线参数
FieldW = "date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg"  #周/月参数 
FieldM = "date,time,code,open,high,low,close,volume,amount,adjustflag"  #分钟参数 
ColBase = ['label', 'industry', 'code', 'code_name']
ColDate = ['year', 'quarter', 'pubDate', 'statDate']
# （财务指标名称，财务指标简称）：(综合指标权重，快速成长，稳健巨头）
ColWs = {('roeAvg','ROE'):        (7,7,7), ('YOYPNI','PYY'):     (7,9,5), 
         ('npMargin','NPM'):      (5,6,4), ('gpMargin','GPM'):   (4,5,3),
         ('YOYEquity','EYY'):     (4,5,3), ('YOYAsset','AYY'):   (3,4,2), 
         ('MBRevenue','MR'):      (2,1,5), ('netProfit','NP'):   (2,1,5), 
         ('AssetTurnRatio','ATR'):(2,2,1), ('NRTurnRatio','RTR'):(2,1,2), ('INVTurnRatio','ITR'):(1,1,1), 
         ('assetToEquity','A2E'): (1,1,2), ('cashRatio','C2L'):  (1,2,1), ('CFOToOR', 'C2R'):    (1,1,1)}
 

# ColProfit = ['code', 'pubDate', 'statDate', 'roeAvg', 'npMargin', 'gpMargin','netProfit', 'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare']
# ColOperation = ['code', 'pubDate', 'statDate', 'NRTurnRatio', 'NRTurnDays','INVTurnRatio', 'INVTurnDays', 'CATurnRatio', 'AssetTurnRatio']
# ColGrowth = ['code', 'pubDate', 'statDate', 'YOYEquity', 'YOYAsset', 'YOYNI','YOYEPSBasic', 'YOYPNI']
# ColBalance = ['code', 'pubDate', 'statDate', 'currentRatio', 'quickRatio','cashRatio', 'YOYLiability', 'liabilityToAsset', 'assetToEquity']
# ColCash = ['code', 'pubDate', 'statDate', 'CAToAsset', 'NCAToAsset','tangibleAssetToAsset', 'ebitToInterest', 'CFOToOR', 'CFOToNP','CFOToGr']
# ColDupont = ['code', 'pubDate', 'statDate', 'dupontROE', 'dupontAssetStoEquity','dupontAssetTurn', 'dupontPnitoni', 'dupontNitogr', 'dupontTaxBurden','dupontIntburden', 'dupontEbittogr']
# Cols = ColProfit[3:]+ColOperation[3:]+ColGrowth[3:]+ColBalance[3:]+ColCash[3:]+ColDupont[3:]
# Weights1 = [1,2,2,1,0,1,0,0]+[2,0,1,0,0,1]+[0.5,0.5,0.5,0.5,0.5]+[1,1,1,0,-1,1]+\
#           [0.5,0,0,0.5,0.5,0.5,0.5]+[0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5]
          

