# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 21:15:51 2018

@author: Administrator
"""
import matplotlib
#matplotlib.use('TkAgg')

from numpy import *
import matplotlib.pyplot as plt

#解析约会数据文件，并将数据导入一个numpy矩阵
def file_parse_matrix(filename):
    with open(filename) as fp:
        Arr_lines = fp.readlines()
        number = len(Arr_lines)
        #初始化数据为m行3列（飞行里程，游戏时间，冰淇淋数）
        #标签单独创建一个向量保存
        return_mat = zeros((number, 3))
        label_vec = []
        index = 0
        
        for line in Arr_lines:
            line = line.strip()
            listFromLine = line.split('\t')  #按换行符分割数据
            #将文本数据前三行存入数据矩阵，第四行存入标签向量
            return_mat[index,:] = listFromLine[0:3]
            label_vec.append(int(listFromLine[3]))
            index += 1
    
    return return_mat, label_vec

