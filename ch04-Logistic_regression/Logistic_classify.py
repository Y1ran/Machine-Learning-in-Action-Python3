# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 15:11:21 2018

@author: Administrator
"""

from numpy import *
from Grad_descent import *
from Random_GDS import Stoch_gdescent

def classifyVec(inp, weights):
    prob = sigmoid(sum(list(array(inp) * array(weights))))
    if prob > 0.5: return 1.0
    else: return 0.0
    
#logistic回归预测算法
def colicTest():
    # 打开训练数据集
    frTrain=open('horseColicTraining.txt')
    # 打开测试数据集
    frTest=open('horseColicTest.txt')

    trainingSet=[];trainingLabels=[]
    # 读取训练集文档的每一行
    for line in frTrain.readlines():
        # 对当前行进行特征分割
        currLine=line.strip().split()
        # 新建列表存储每个样本的特征向量
        lineArr=[]
        # 遍历每个样本的特征
        for i in range(21):
            # 将该样本的特征存入lineArr列表
            lineArr.append(float(currLine[i]))
        #将该样本标签存入标签列表
        trainingLabels.append(currLine[21])
        #将该样本的特征向量添加到数据集列表
        trainingSet.append(lineArr)
    #调用随机梯度上升法更新logistic回归的权值参数
    trainWeights=Stoch_gdescent(trainingSet,trainingLabels,500)
    #统计测试数据集预测错误样本数量和样本总数
    errorCount=0; numTestVec=0.0
    #遍历测试数据集的每个样本
    for line in frTest.readlines():
        #样本总数加1
        numTestVec+=1.0
        #对当前行进行处理，分割出各个特征及样本标签
        currLine=line.strip().split()
        #新建特征向量
        lineArr=[]
        #将各个特征构成特征向量
        for i in range(21):
            lineArr.append(float(currLine[i]))
        #利用分类预测函数对该样本进行预测，并与样本标签进行比较
        if(classifyVec(lineArr,trainWeights)!=currLine[21]):
            #如果预测错误，错误数加1
            errorCount+=1
    #计算测试集总的预测错误率
    errorRate=(float(errorCount)/numTestVec)
    #打印错误率大小
    print('the error rate of this test is: %f' % (errorRate))    
    #返回错误率
    return errorRate

#多次测试算法求取预测误差平均值
def multTest():
    #设置测试次数为10次，并统计错误率总和
    numTests=10;errorRateSum=0.0
    #每一次测试算法并统计错误率
    for k in range(numTests):
        errorRateSum+=colicTest()
    #打印出测试10次预测错误率平均值
    print('after %d iterations the average error rate is: %f' \
     % (numTests,errorRateSum / float(numTests)))