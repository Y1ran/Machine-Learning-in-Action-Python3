# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:05:47 2018

@author: Administrator
"""

#收集数据
#添加时间函数库
from time import sleep
#添加json库
import json
#添加urllib2库
import requests
#@retX:样本玩具特征矩阵
#@retY：样本玩具的真实价格
#@setNum：获取样本的数量
#@yr：样本玩具的年份
#@numPce:玩具套装的零件数
#@origPce:原始价格
def searchForSet(retX,retY,setNum,yr,numPce,origPrc):
    #睡眠十秒
    sleep(10)
    #拼接查询的url字符串
    myAPIstr='get from code.google.com'
    searchURL='https://www.googleapis.com/shopping/search/v1/public/products?\
        key=%s&country=US&q=lego+%d&alt=json' %(myAPIstr,setNum)
    #利用urllib2访问url地址
    pg=requests.urlopen(searchURL)
    #利用json打开和解析url获得的数据，数据信息存入字典中
    retDict=json.load(pg.read())
    #遍历数据的每一个条目
    for i in range(len(retDict['items'])):
        try:
            #获得当前条目
            currItem=retDict['items'][i]
            #当前条目对应的产品为新产品
            if currItem['product']['condition']=='new':
                #标记为新
                newFlag=1
            else:newFlag=0
            #得到当前目录产品的库存列表
            listOfInv=currItem['product']['inventories']
            #遍历库存中的每一个条目
            for item in listOfInv:
                #得到该条目玩具商品的价格
                sellingPrice=item['price']
                #价格低于原价的50%视为不完整套装
                if sellingPrice> (origPrc*0.5):
                    print('%d\t%d\t%d\t%f\t%f' %(yr,numPce,newFlag,\
                        origPrc,sellingPrice))
                    #将符合条件套装信息作为特征存入数据矩阵
                    retX.append([yr,numPce,newFlag,origPce])
                    #将对应套装的出售价格存入矩阵
                    retY.append(sellingPrice)
        except:print('problem with item %d' % i)

#多次调用收集数据函数，获取多组不同年份，不同价格的数据
def setDataCollect(retX,retY):
    searchForSet(retX,retY,8288,2006,800,49.99)
    searchForSet(retX,retY,10030,2002,3096,49.99)
    searchForSet(retX,retY,10179,2007,5195,499.99)
    searchForSet(retX,retY,10181,2007,3428,199.99)
    searchForSet(retX,retY,10189,2008,5922,299.99)
    searchForSet(retX,retY,10196,2009,3263,249.99)