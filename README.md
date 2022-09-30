# Machine-Learning-in-Action-Python3 
<br>

这个月开始练习《机器学习实战》，原书比较久远了，且代码和练习都是基于Python2，我个人是升级到了Python3，因此使用最新的版本来写这些习题。具体2和3其实在基础语法上并没有太多差别，一些高级特性比如装饰器工厂，协程，IO等Python3的新用法，一般机器学习也用不上，因为追求性能的话都会用C/C++等语言去实现，Python只是小规模的测试用。<br>

目前刚开始做，有不对的欢迎指正，也欢迎大家star。除了 版本差异，代码里的部分函数以及代码范式也和原书不一样（因为作者的代码实在让人看的别扭，我改过后看起来舒服多了）。在这个系列之后，我还会写一个scikit-learn机器学习系列，因为在实现了源码之后，带大家看看SK框架如何使用也是非常重要的。

# 什么是K-近邻算法？

简单地说，k-近邻算法采用测量不同特征值之间距离的方法进行分类。不恰当但是形象地可以表述为近朱者赤，近墨者黑。它有如下特点：
<br>

优点：精度高、对异常值不敏感、无数据输入假定
缺点：计算复杂度高、空间复杂度高
适用数据范围：数值型和标称型
<br>
# K-近邻算法的工作原理

<br>存在一个样本数据集合，也称作训练样本集，并且样本集中的每个数据都存在标签，即我们知道样本集中每一数据与所属分类的对应关系。输入没有标签的数据后，将这个没有标签的数据的每个特征与样本集中的数据对应的特征进行比较，然后算法提取样本中特征最相似的数据（最邻近）的分类标签。一般来说，我们只选择样本数据集中前K个最相似的数据，这就是K近邻算法中K的出处，通常是不大于 20 的整数。最后，选择K个最相似数据中出现次数最多的类别，作为新数据的分类。<br>

<br>我们想使用K近邻算法来分来爱情片和动作片。有人曾统计过很多电影的打斗镜头和接吻镜头，下图显示了 6 部电影的打斗镜头和接吻镜头数。假如有一部未看过的电影，如何确定它是爱情片还是动作片呢？（当然了，我们这里不考虑爱情动作片）我们可以使用 kNN(k-nearest neighbors algorithm) 来解决这个问题。
<br>

# K-近邻算法的一般流程

搜集数据：可以使用任何方法。
准备数据：距离计算所需要的值，最好是结构化的数据。
分析数据：可以使用任何方法。
训练算法：此步骤不适用于-近邻算法。
测试算法：计算错误率。
使用算法：首先需要输入样本数据和待分类数据，然后运行-近邻算法判定待分类数据分别属于哪个分类，最后应用计算出的分类执行后续的处理。
