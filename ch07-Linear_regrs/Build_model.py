# -*- coding: utf-8 -*-
"""
Created on Sat Jul 14 15:06:05 2018

@author: Administrator
"""

#训练算法：建立模型
#交叉验证测试岭回归
#@xArr:从网站中获得的玩具套装样本数据
#@yArr：样本对应的出售价格
#@numVal:交叉验证次数
def crossValidation(xArr,yArr,numVal=10):
    #m,n=shape(xArr)
    #xArr1=mat(ones((m,n+1)))
    #xArr1[:,1:n+1]=mat(xArr)
    #获取样本数
    m=len(yArr)
    indexList=range(m)
    #将每个回归系数对应的误差存入矩阵
    errorMat=zeros((numVal,30))
    #进行10折交叉验证
    for i in range(numVal):
        trainX=[];trainY=[]
        testX=[];testY=[]
        #混洗索引列表
        random.shuffle(indexList)
        #遍历每个样本
        for j in range(m):
            #数据集90%作为训练集
            if j<m*0.9:
                trainX.append(xArr1[indexList[j]])
                trainY.append(yArr[indexList[j]])
            #剩余10%作为测试集
            else:
                testX.append(xArr1[indexList[j]])
                testY.append(yArr[indexList[j]])
        #利用训练集计算岭回归系数
        wMat=ridgeRegres(trainX,trainY)
        #对于每一个验证模型的30组回归系数
        for k in range(30):
            #转为矩阵形式
            matTestX=mat(testX);matTrainX=mat(trainX)
            #求训练集特征的均值
            meanTrain=mean(matTrainX,0)
            #计算训练集特征的方差
            varTrain=val(matTrainX,0)
            #岭回归需要对数据特征进行标准化处理
            #测试集用与训练集相同的参数进行标准化
            matTestX=(matTestX-meanTrain)/varTrain
            #对每组回归系数计算测试集的预测值
            yEst=matTestX*mat(wMat[k,:]).T+mean(trainY)
            #将原始值和预测值的误差保存
            errorMat[i,k]=rssError(yEst.T.A,array(testY))
    #对误差矩阵中每个lamda对应的10次交叉验证的误差结果求均值
    meanErrors=mean(errorMat,0)
    #找到最小的均值误差
    minMean=float(min(meanErrors))
    #将均值误差最小的lamda对应的回归系数作为最佳回归系数
    bestWeigths=wMat[nonzero(meanErrors==minMean)]
    xMat=mat(xArr);yMat=mat(yArr).T
    meanX=mean(xMat,0);valX=val(xMat,0)
    #数据标准化还原操作
    unReg=bestWeigths/valX
    print('the best model from Ridge Regression is :\n',unReg)
    print('with constant term :',-1*sum(multiply(meanX,unReg))+mean(yMat))