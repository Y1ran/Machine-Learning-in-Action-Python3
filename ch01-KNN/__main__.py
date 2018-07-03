# -*- coding: utf-8 -*-
"""
Created on Mon Jul  2 21:18:57 2018

@author: Administrator
"""

#from numpy import *
#from Digit_recog import *


if __name__ == '__main__':
    # 测试数据
    group, labels = createDataSet()
    classify_KNN([0,0], group, labels, 3)
    
    DataMat, LabelMat = file_parse_matrix('datingTestSet2.txt')
    print(DataMat,shape(DataMat),LabelMat)
    
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(DataMat[:,1],DataMat[:,2])
    plt.show()
    
    dating_mat, label_mat = file_parse_matrix('datingTestSet2.txt')
    data_normed, ranges, minV = Norm_feature(dating_mat)
    Test_accuray(0.1, dating_mat, label_mat)
    
    testVec = img2vec('digits/testDigits/0_13.txt')
    print(testVec)
    
    # HandWritingTest('digits/trainingDigits', 'digits/testDigits/')
    # 这行代码耗时比较久，可以单独运行