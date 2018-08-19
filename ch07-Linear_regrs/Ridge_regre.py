# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:04:10 2018

@author: Administrator
"""


def ridgeRegres(xMat,yMat,lam=0.2):
    '''
    #岭回归
    @xMat:样本数据
    @yMat：样本对应的原始值
    @lam：惩罚项系数lamda，默认值为0.2
    '''
    #计算矩阵内积
    xTx=xMat.T*xMat
    #添加惩罚项，使矩阵xTx变换后可逆
    denom=xTx+eye(shape(xMat)[1])*lam
    #判断行列式值是否为0，确定是否可逆
    if linalg.det(denom)==0.0:
        print('This matrix is singular,cannot do inverse')
        return 
    #计算回归系数
    ws=denom.I*(xMat.T*yMat)
    return ws

#特征需要标准化处理，使所有特征具有相同重要性
def ridgeTest(xArr,yArr):
    xMat=mat(xArr);yMat=mat(yArr).T
    #计算均值
    yMean=mean(yMat,0)
    yMat=yMat-yMean
    xMeans=mean(xMat,0)
    #计算各个特征的方差
    xVar=var(xMat,0)
    #特征-均值/方差
    xMat=(xMat-xMeans)/xVar
    #在30个不同的lamda下进行测试
    numTestpts=30
    #30次的结果保存在wMat中
    wMat=zeros((numTestpts,shape(xMat)[1]))
    for i in range(numTestpts):
        #计算对应lamda回归系数，lamda以指数形式变换
        ws=ridgeRegres(xMat,yMat,exp(i-10))
        wMat[i,:]=ws.T
    return wMat