# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:26:21 2018

@author: Administrator
"""

from numpy import *
from Kernel import *

#启发式SMO算法的支持函数
#新建一个类的收据结构，保存当前重要的值
class optStruct:
    def __init__(self,dataMatIn,classLabels,C,toler, kTup):
        self.X=dataMatIn
        self.labelMat=classLabels
        self.C=C
        self.tol=toler
        self.m=shape(dataMatIn)[0]
        self.alphas=mat(zeros((self.m,1)))
        self.b=0
        self.Cache=mat(zeros((self.m,2)))
        self.K = mat(zeros((self.m,self.m)))
        for i in range(self.m):
            self.K[:,i] = kernelTrans(self.X, self.X[i,:], kTup)
#格式化计算误差的函数，方便多次调用


def calcEk(oS, k):
    '''计算预测误差'''
    fXk = float(multiply(oS.alphas,oS.labelMat).T*oS.K[:,k] + oS.b)
    Ek = fXk - float(oS.labelMat[k])
    return Ek
#修改选择第二个变量alphaj的方法
def selectJ(i,oS,Ei):
    maxK=-1;maxDeltaE= 0 ;Ej=0
    #将误差矩阵每一行第一列置1，以此确定出误差不为0
    #的样本
    oS.Cache[i]=[1,Ei]
    #获取缓存中Ei不为0的样本对应的alpha列表
    validEcacheList=nonzero(oS.Cache[:,0].A)[0]
    #在误差不为0的列表中找出使abs(Ei-Ej)最大的alphaj
    if(len(validEcacheList)>0):
        for k in validEcacheList:
            if k ==i:continue
            Ek=calcEk(oS,k)
            deltaE=abs(Ei-Ek)
            if(deltaE>maxDeltaE):
                maxK=k;maxDeltaE=deltaE;Ej=Ek
        return maxK,Ej
    else:
    #否则，就从样本集中随机选取alphaj
        j=selectJrand(i,oS.m)
        Ej=calcEk(oS,j)
    return j,Ej
#更新误差矩阵
def updateEk(oS,k):
    Ek=calcEk(oS,k)
    oS.Cache[k]=[1,Ek]