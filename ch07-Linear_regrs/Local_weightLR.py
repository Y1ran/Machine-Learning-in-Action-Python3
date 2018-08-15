# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:03:07 2018

@author: Administrator
"""

#局部加权线性回归
def rssError(yArr,yHatArr):
    #返回平方误差和
    return ((yArr-yHatArr)**2).sum()


def lwlr(testPoint,xArr,yArr,k=1.0):
    '''
    #每个测试点赋予权重系数
    @testPoint:测试点
    @xArr：样本数据矩阵
    @yArr：样本对应的原始值
    @k：用户定义的参数，决定权重的大小，默认1.0
    '''
    #转为矩阵形式
    xMat=mat(xArr);yMat=mat(yArr).T
    #样本数
    m=shape(xMat)[0]
    #初始化权重矩阵为m*m的单位阵
    weights=mat(eye((m)))
    #循环遍历各个样本
    for j in range(m):
        #计算预测点与该样本的偏差
        diffMat=testPoint-xMat[j,:]
        #根据偏差利用高斯核函数赋予该样本相应的权重
        weights[j,j]=exp(diffMat*diffMat.T/(-2.0*k**2))
    #将权重矩阵应用到公式中
    xTx=xMat.T*(weights*xMat)
    #计算行列式值是否为0，即确定是否可逆
    if linalg.det(xTx)==0.0:
        print('This matrix is singular,cannot do inverse')
        return 
    #根据公式计算回归系数
    ws=xTx.I*(xMat.T*(weights*yMat))
    #计算测试点的预测值
    return testPoint*ws

#测试集进行预测    
def lwlrTest(testArr,xArr,yArr,k=1.0):
    #测试集样本数
    m=shape(testArr)[0]
    #测试集预测结果保存在yHat列表中
    yHat=zeros(m)
    #遍历每一个测试样本
    for i in range(m):
        #计算预测值
        yHat[i]=lwlr(testArr[i],xArr,yArr,k)
    return yHat