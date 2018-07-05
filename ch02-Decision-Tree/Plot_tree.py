# -*- coding: utf-8 -*-
"""
Created on Wed Jul  4 21:10:10 2018

@author: Administrator
"""

import matplotlib.pyplot as plt

# 定义文本框和箭头格式
decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")
 
# 绘制带箭头的注释
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
     createPlot.ax1.annotate(nodeTxt, xy=parentPt,  xycoords='axes fraction',
              xytext=centerPt, textcoords='axes fraction',
              va="center", ha="center", bbox=nodeType, arrowprops=arrow_args )
 
     
def Num_of_leaf(myTree):
    '''计算此树的叶子节点数目'''
    num_leaf = 0
    first_node = myTree.keys()
    first_node = list(first_node)[0]
    second_dict = myTree[first_node]
    # Python3中使用LIST转换firstnode，原书使用[0]直接索引只能用于Python2
    # 对于树，每次判断value是否为字典，若为字典则进行递归，否则累加器+1
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ =='dict':
            num_leaf += Num_of_leaf(second_dict[key])
        else: num_leaf += 1
    return num_leaf

def Depth_of_tree(myTree):
    '''计算此树的总深度'''
    depth = 0
    first_node = myTree.keys()
    first_node = list(first_node)[0]
    second_dict = myTree[first_node]
    
    for key in second_dict.keys():
        if type(second_dict[key]).__name__ =='dict':
            pri_depth = 1 + Depth_of_tree(second_dict[key])
        else: pri_depth = 1
        # 对于树，每次判断value是否为字典，若为字典则进行递归，否则计数器+1
        if pri_depth > depth: depth = pri_depth
    return depth

def retrieveTree(i):
    '''
   保存了树的测试数据
     '''
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': {0: 'no', \
                                1: 'yes'}}}},{'no surfacing': {0: 'no', \
    1: {'flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]

def plotmidtext(cntrpt,parentpt,txtstring):
    '''作用是计算tree的中间位置    
    cntrpt起始位置,parentpt终止位置,txtstring：文本标签信息
    '''
    xmid=(parentpt[0]-cntrpt[0])/2.0+cntrpt[0]
    # cntrPt 起点坐标 子节点坐标   
    # parentPt 结束坐标 父节点坐标
    ymid=(parentpt[1]-cntrpt[1])/2.0+cntrpt[1] # 找到x和y的中间位置
    createPlot.ax1.text(xmid,ymid,txtstring)
    
    
def plottree(mytree,parentpt,nodetxt):
    numleafs=Num_of_leaf(mytree)
    depth=Depth_of_tree(mytree)
    firststr=list(mytree.keys())[0]
    cntrpt=(plottree.xoff+(1.0+float(numleafs))/2.0/plottree.totalw,plottree.yoff)
    # 计算子节点的坐标 
    plotmidtext(cntrpt,parentpt,nodetxt) #绘制线上的文字  
    plotNode(firststr,cntrpt,parentpt,decisionNode)#绘制节点  
    seconddict=mytree[firststr]
    plottree.yoff=plottree.yoff-1.0/plottree.totald
    # 每绘制一次图，将y的坐标减少1.0/plottree.totald，间接保证y坐标上深度的
    for key in seconddict.keys():
        if type(seconddict[key]).__name__=='dict':
            plottree(seconddict[key],cntrpt,str(key))
        else:
            plottree.xoff=plottree.xoff+1.0/plottree.totalw
            plotNode(seconddict[key],(plottree.xoff,plottree.yoff),cntrpt,leafNode)
            plotmidtext((plottree.xoff,plottree.yoff),cntrpt,str(key))
    plottree.yoff=plottree.yoff+1.0/plottree.totald
 
    
def createPlot(intree):
    # 类似于Matlab的figure，定义一个画布(暂且这么称呼吧)，背景为白色 
    fig=plt.figure(1,facecolor='white')
    fig.clf()    # 把画布清空 
    axprops=dict(xticks=[],yticks=[])   
    # createPlot.ax1为全局变量，绘制图像的句柄，subplot为定义了一个绘图，
    # 111表示figure中的图有1行1列，即1个，最后的1代表第一个图 
    # frameon表示是否绘制坐标轴矩形 
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops) 
    
    plottree.totalw=float(Num_of_leaf(intree))
    plottree.totald=float(Depth_of_tree(intree))
    plottree.xoff=-0.6/plottree.totalw;plottree.yoff=1.2;
    plottree(intree,(0.5,1.0),'')
    plt.show()
