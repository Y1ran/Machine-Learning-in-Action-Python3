# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:05:20 2018

@author: Administrator
"""

#前向逐步回归
#@eps：每次迭代需要调整的步长    
def stageWise(xArr,yArr,eps=0.01,numIt=100):
    xMat=mat(xArr);yMat=mat(yArr)
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    #将特征标准化处理为均值为0，方差为1
    xMat=regularize(xMat)
    m,n=shape(xMat)
    #将每次迭代中得到的回归系数存入矩阵
    returnMat=zeros((numIt,m))
    ws=zeros((n,1));wsTest=ws.copy();wsMat=ws.copy()
    for i in range(numIt):
        print(ws.T)
        #初始化最小误差为正无穷
        lowestError=inf;
        for j in range(n):
            #对每个特征的系数执行增加和减少eps*sign操作
            for sign in [-1,1]:
                wsTest=ws.copy()
                wsTest[j]+=eps*sign
                #变化后计算相应预测值
                yTest=xMat*wsTest
                #保存最小的误差以及对应的回归系数
                rssE=rssError(yMat.A,yTest.A)
                if rssE<lowestError:
                    lowestError=rssE
                    wsMat=wsTest
        ws=wsMat.copy()
        returnMat[i,:]=ws
    return returnMat