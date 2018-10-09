import numpy as np
import os
from PIL import Image
import operator

import matplotlib.pyplot as plt
import sys


#直接把print保存在TXT文件中，这样我就可以不用复制黏贴了哈哈哈！
class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "a")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


def get_files(file_dir):
    c0 = []
    label_c0 = []
    c1 = []
    label_c1 = []
    c2 = []
    label_c2 = []
    c3 = []
    label_c3 = []
    c4 = []
    label_c4 = []
    c5 = []
    label_c5 = []
    c6 = []
    label_c6 = []
    c7 = []
    label_c7 = []
    c8 = []
    label_c8 = []
    c9 = []
    label_c9 = []

    for file in os.listdir(file_dir+'/c0'):
        c0.append(file_dir+'/c0'+'/'+file)
        label_c0.append(0)
    for file in os.listdir(file_dir+'/c1'):
        c1.append(file_dir+'/c1'+'/'+file)
        label_c1.append(1)
    for file in os.listdir(file_dir+'/c2'):
        c2.append(file_dir+'/c2'+'/'+file)
        label_c2.append(2)
    for file in os.listdir(file_dir+'/c3'):
        c3.append(file_dir+'/c3'+'/'+file)
        label_c3.append(3)
    for file in os.listdir(file_dir+'/c4'):
        c4.append(file_dir+'/c4'+'/'+file)
        label_c4.append(4)
    for file in os.listdir(file_dir+'/c5'):
        c5.append(file_dir+'/c5'+'/'+file)
        label_c5.append(5)
    for file in os.listdir(file_dir+'/c6'):
        c6.append(file_dir+'/c6'+'/'+file)
        label_c6.append(6)
    for file in os.listdir(file_dir+'/c7'):
        c7.append(file_dir+'/c7'+'/'+file)
        label_c7.append(7)
    for file in os.listdir(file_dir+'/c8'):
        c8.append(file_dir+'/c8'+'/'+file)
        label_c8.append(8)
    for file in os.listdir(file_dir+'/c9'):
        c9.append(file_dir+'/c9'+'/'+file)
        label_c9.append(9)
    print('There are %d c0\nThere are %d c1\nThere are %d c2\nThere are %d c3\nThere are %d c4\nThere are %d c5\nThere are %d c6\nThere are %d c7\nThere are %d c8\nThere are %d c9' % (len(c0), len(c1), len(c2), len(c3),len(c4), len(c5), len(c6), len(c7),len(c8), len(c9)))

    image_list = np.hstack((c0, c1, c2, c3, c4, c5, c6, c7, c8, c9))
    label_list = np.hstack((label_c0, label_c1, label_c2, label_c3, label_c4, label_c5, label_c6, label_c7, label_c8, label_c9))
    # 利用shuffle打乱顺序
    temp = np.array([image_list, label_list])
    temp = temp.transpose()
    np.random.shuffle(temp)
    # # 从打乱的temp中再取出list（img和lab）
    image_list = list(temp[:, 0])
    label_list = list(temp[:, 1])
    label_list = [int(i) for i in label_list]

    return image_list, label_list


def classify0(inX, dataSet, labels, k):
    """
    Desc:
        kNN 的分类函数
    Args:
        inX -- 用于分类的输入向量/测试数据
        dataSet -- 训练数据集的 features
        labels -- 训练数据集的 labels
        k -- 选择最近邻的数目
    Returns:
        sortedClassCount[0][0] -- 输入向量的预测分类 labels

    注意：labels元素数目和dataSet行数相同；程序使用欧式距离公式.

    预测数据所在分类可在输入下列命令
    kNN.classify0([0,0], group, labels, 3)
    """

    # -----------实现 classify0() 方法的第一种方式----------------------------------------------------------------------------------------------------------------------------
    # 1. 距离计算
    dataSetSize = dataSet.shape[0]
    # tile生成和训练样本对应的矩阵，并与训练样本求差
    """
    tile: 列-3表示复制的行数， 行-1／2表示对inx的重复的次数

    In [8]: tile(inx, (3, 1))
    Out[8]:
    array([[1, 2, 3],
        [1, 2, 3],
        [1, 2, 3]])

    In [9]: tile(inx, (3, 2))
    Out[9]:
    array([[1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3],
        [1, 2, 3, 1, 2, 3]])
    """
    diffMat = np.tile(inX, (dataSetSize, 1)) - dataSet
    """
    欧氏距离： 点到点之间的距离
       第一行： 同一个点 到 dataSet 的第一个点的距离。
       第二行： 同一个点 到 dataSet 的第二个点的距离。
       ...
       第N行： 同一个点 到 dataSet 的第N个点的距离。

    [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
    (A1-A2)^2+(B1-B2)^2+(c1-c2)^2
    """
    # 取平方
    sqDiffMat = diffMat ** 2
    # 将矩阵的每一行相加
    sqDistances = sqDiffMat.sum(axis=1)
    # 开方
    distances = sqDistances ** 0.5
    # 根据距离排序从小到大的排序，返回对应的索引位置
    # argsort() 是将x中的元素从小到大排列，提取其对应的index（索引），然后输出到y。
    # 例如：x=np.array([1,4,3,-1,6,9]),y=array([3,0,2,1,4,5]) 则，x[3]=-1最小，所以y[0]=3;x[5]=9最大，所以y[5]=5。
    # print 'distances=', distances
    sortedDistIndicies = distances.argsort()
    # print 'distances.argsort()=', sortedDistIndicies

    # 2. 选择距离最小的k个点
    classCount = {}
    for i in range(k):
        # 找到该样本的类型
        voteIlabel = labels[sortedDistIndicies[i]]
        # 在字典中将该类型加一
        # 字典的get方法
        # 如：list.get(k,d) 其中 get相当于一条if...else...语句,参数k在字典中，字典将返回list[k];如果参数k不在字典中则返回参数d,如果K在字典中则返回k对应的value值
        # l = {5:2,3:4}
        # print l.get(3,0)返回的值是4；
        # Print l.get（1,0）返回值是0；
        classCount[voteIlabel] = classCount.get(voteIlabel, 0) + 1
    # 3. 排序并返回出现最多的那个类型
    # 字典的 items() 方法，以列表返回可遍历的(键，值)元组数组。
    # 例如：dict = {'Name': 'Zara', 'Age': 7}   print "Value : %s" %  dict.items()   Value : [('Age', 7), ('Name', 'Zara')]
    # sorted 中的第2个参数 key=operator.itemgetter(1) 这个参数的意思是先比较第几个元素
    # 例如：a=[('b',2),('a',1),('c',0)]  b=sorted(a,key=operator.itemgetter(1)) >>>b=[('c',0),('a',1),('b',2)] 可以看到排序是按照后边的0,1,2进行排序的，而不是a,b,c
    # b=sorted(a,key=operator.itemgetter(0)) >>>b=[('a',1),('b',2),('c',0)] 这次比较的是前边的a,b,c而不是0,1,2
    # b=sorted(a,key=opertator.itemgetter(1,0)) >>>b=[('c',0),('a',1),('b',2)] 这个是先比较第2个元素，然后对第一个元素进行排序，形成多级排序,reverse = True降序。
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

    # ------------------------------------------------------------------------------------------------------------------------------------------
    # 实现 classify0() 方法的第二种方式

    # """
    # 1. 计算距离

    # 欧氏距离： 点到点之间的距离
    #    第一行： 同一个点 到 dataSet的第一个点的距离。
    #    第二行： 同一个点 到 dataSet的第二个点的距离。
    #    ...
    #    第N行： 同一个点 到 dataSet的第N个点的距离。

    # [[1,2,3],[1,2,3]]-[[1,2,3],[1,2,0]]
    # (A1-A2)^2+(B1-B2)^2+(c1-c2)^2

    # inx - dataset 使用了numpy broadcasting，见 https://docs.scipy.org/doc/numpy-1.13.0/user/basics.broadcasting.html
    # np.sum() 函数的使用见 https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.sum.html
    # """


if __name__=='__main__':
    #把print写入到txt
    sys.stdout = Logger("result.txt")

    data_dir = './img'
    data, data_label = get_files(data_dir)
    num = int(len(data) / 2)
    data_1 = data[0:num]
    data_label_1 = data_label[0:num]
    data_2 = data[num:len(data)+1]
    data_label_2 = data_label[num:len(data)+1]
    #第一次交叉验证
    m = len(data_label_1)
    trainingMat = np.zeros((m, 1024)) #32*32
    for i in range(m):
        image = Image.open(data_1[i])
        image = np.array(image.resize((32,32)))
        image = image.reshape((1,-1))
        trainingMat[i] = image
        # plt.imshow(image)
        # plt.show()
    errorCount_1 = 0
    mTest_1 = len(data_label_2)
    for i in range(mTest_1):
        image = Image.open(data_2[i])
        image = np.array(image.resize((32, 32)))
        image = image.reshape((1, -1))
        classifierResult = classify0(image, trainingMat, data_label_1, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, data_label_2[i]))
        errorCount_1 += classifierResult != data_label_2[i]

    #第二次交叉验证
    m = len(data_label_2)
    trainingMat = np.zeros((m, 1024)) #32*32
    for i in range(m):
        image = Image.open(data_2[i])
        image = np.array(image.resize((32,32)))
        image = image.reshape((1,-1))
        trainingMat[i] = image
        # plt.imshow(image)
        # plt.show()
    errorCount_2 = 0
    mTest_2 = len(data_label_1)
    for i in range(mTest_2):
        image = Image.open(data_1[i])
        image = np.array(image.resize((32, 32)))
        image = image.reshape((1, -1))
        classifierResult = classify0(image, trainingMat, data_label_2, 3)
        print("the classifier came back with: %d, the real answer is: %d" % (classifierResult, data_label_1[i]))
        errorCount_2 += classifierResult != data_label_1[i]
    # print("\nthe half number of errors is: %d" % errorCount_1)
    # print("\nthe half error rate is: %f" % (errorCount_1 / mTest_1))
    # print("\nthe half number of errors is: %d" % errorCount_2)
    # print("\nthe half error rate is: %f" % (errorCount_2 / mTest_2))
    print("\nthe total number of errors is: %d" % (errorCount_2+errorCount_1))
    print("\nthe total error rate is: %f" % ((errorCount_2+errorCount_1) / (mTest_1+mTest_2)))

