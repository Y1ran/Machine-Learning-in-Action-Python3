# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 13:51:17 2018

@author: Administrator
"""

import matplotlib
import matplotlib.pyplot as plt
from numpy import *

def plot_fit(data, labelMat, weights):
    dataArr = array(data)
    n = shape(dataArr)[0]
    
    x_cord1 = []; y_cord1 = []
    x_cord2 = []; y_cord2 = []
    for i in range(n):  
        if int(labelMat[i]) == 1:  
            x_cord1.append(dataArr[i,1]); y_cord1.append(dataArr[i,2])  
        else: x_cord2.append(dataArr[i,1]); y_cord2.append(dataArr[i,2])  
    
    fig = plt.figure()
    ax = fig.add_subplot(111)  
    ax.scatter(x_cord1, y_cord1, s = 30, c = 'red', marker='s')  
    ax.scatter(x_cord2, y_cord2, s = 30, c = 'green')  
    
    x = arange(-3.0, 3.0, 0.1)  
    y = ((-weights[0]- weights[1] * x)/weights[2]).transpose()
    ax.plot(x, y)  
    plt.xlabel('X1');  
    plt.ylabel('X2');  
    plt.show()  