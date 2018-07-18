# -*- coding: utf-8 -*-
"""
Created on Tue Jul 10 12:00:50 2018

@author: Administrator
"""

import random

# SMO算法相关辅助中的辅助函数
# 解析文本数据函数，提取每个样本的特征组成向量，添加到数据矩阵
# 添加样本标签到标签向量
def loadDataSet(filename):
    dataMat=[];labelMat=[]
    fr=open(filename)
    
    for line in fr.readlines():
        lineArr=line.strip().split('\t')
        dataMat.append([float(lineArr[0]),float(lineArr[1])])
        #if int(lineArr[2]) == 0 :
        # labelMat.append((float(lineArr[2]) - 1))
        #else:
        labelMat.append((float(lineArr[2] )))
    return dataMat,labelMat

#2 在样本集中采取随机选择的方法选取第二个不等于第一个alphai的
#优化向量alphaj
def selectJrand(i,m):
    j=i
    while(j==i):
        j=int(random.uniform(0,m))
    return j

#3 约束范围L<=alphaj<=H内的更新后的alphaj值    
def clipAlpha(aj,H,L):
    if aj > H:
        aj = H
    if L > aj:
        aj = L
    return aj