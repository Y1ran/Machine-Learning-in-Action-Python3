# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:02:40 2018

@author: Administrator
"""

from numpy import *

#解析文件中的数据为适合机器处理的形式
def loadDataSet(filename):
    numFeat=len(open(filename).readline().split('\t'))-1
    dataMat=[];labelMat=[]
    fr=open(filename)
    for line in fr.readlines():
        lineArr=[]
        curLine=line.strip().split('\t')
        for i in range(numFeat):
            lineArr.extend(float(curLine[i]))
        dataMat.append(lineArr)
        labelMat.append(float(curLine[-1]))
    return dataMat,labelMat

#标准线性回归算法
#ws=(X.T*X).I*(X.T*Y)    
def standRegres(xArr,yArr):
    #将列表形式的数据转为numpy矩阵形式
    xMat=mat(xArr);yMat=mat(yArr).T
    #求矩阵的内积
    xTx=xMat.T*xMat
    #numpy线性代数库linalg
    #调用linalg.det()计算矩阵行列式
    #计算矩阵行列式是否为0
    if linalg.det(xTx)==0.0:
        print('This matrix is singular,cannot do inverse')
        return 
    #如果可逆，根据公式计算回归系数
    ws=xTx.I*(xMat.T*yMat)
    #可以用yHat=xMat*ws计算实际值y的预测值
    #返回归系数
    return ws