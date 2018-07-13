# -*- coding: utf-8 -*-
"""
Created on Sun Jul  8 13:05:29 2018

@author: Administrator
"""

from numpy import *
from Grad_descent import *
from Plot_boundary import *
from matplotlib import *
from Random_GDS import *
from Logistic_classify import *

if __name__ == '__main__':
    data, label = loadData('testSet.txt')
    print(Stoch_gdescent(data, label))
    
    weights = Stoch_gdescent(data, label)
    plot_fit(data, label, weights)
    
    multTest()