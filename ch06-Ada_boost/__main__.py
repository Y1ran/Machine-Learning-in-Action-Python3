# -*- coding: utf-8 -*-
"""
Created on Fri Jul 13 12:21:58 2018

@author: Administrator
"""
from numpy import *
from Stump_classify import *
from Test_classify import *
from Train_adaboost import *
from ROC_plot import *


def loadSimpData():
    dataMat=matrix([[1. ,2.1],
        [2. ,1.1],
        [1.3,1. ],
        [1. ,1. ],
        [2. ,1. ]])
    classLabels=[1.0,1.0,-1.0,-1.0,1.0]
    return dataMat,classLabels

if __name__ == '__main__':
    data, label = loadSimpData()
    datArr,labelArr = loadDataSet('horseColicTraining.txt')
    classifierArr = adaBoostTrainDS(datArr,labelArr,9)
    testArr,testLabelArr = loadDataSet('horseColicTest.txt')
    prediciton = adaClassify(testArr,classifierArr)

    error = mat(ones((67,1)))
    error[prediciton != mat(testLabelArr ).T] .sum() 