# -*- coding: utf-8 -*-
"""
Created on Tue Jul  3 12:06:45 2018

@author: Administrator
"""

from os import listdir
from numpy import *
from KNN_algr import *

def img2vec(filename):
    '''this is to...将32X32的图像转化为1X1024的行向量'''
    returnvec = zeros((1,1024))
    
    with open(filename) as fp:
        for i in range(32):
            line = fp.readline()
            for j in range(32):
                returnvec[0, 32*i + j] = int(line[j])
    # returnVEC按32进位，j代表每位的32个元素    
    return returnvec

def HandWritingTest(train_dir,test_dir):
    labels = []
    File_list = listdir(train_dir)
    # 将目录内的文件按名字放入列表，使用函数解析为数字
    m = len(File_list)
    train_mat = zeros((m,1024))
    for i in range(m):
        fname = File_list[i]
        fstr = fname.split('.')[0]
        fnumber = int(fstr.split('_')[0])
    # 比如'digits/testDigits/0_13.txt'，被拆分为0,13,txt
    # 此处0即为标签数字   
        labels.append(fnumber)
        train_mat[i,:] = img2vec('%s/%s' % (train_dir,fname))
    # labels is label_vec，同之前的KNN代码相同，存储标签
    
    test_File_list = listdir(test_dir)
    error = 0.0
    test_m = len(test_File_list)
    for i in range(test_m):
        fname = test_File_list[i]
        fstr = fname.split('.')[0]
        fnumber = int(fstr.split('_')[0])
        vec_test = img2vec('digits/testDigits/%s' % fname)
        Result = classify_KNN(vec_test, train_mat, labels, 3)
        print("the classifier came with: %d, the real answer is :%d " \
                 % (Result, fnumber))
        if(Result != fnumber) : error += 1
    # 这部分和Test模块相同，直接copy过来就好
    print("the accuracy is %f | the error_rate is %f " % \
          (1- (float(error) /float(test_m)),(float(error) /float(test_m))))