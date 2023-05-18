#! /usr/bin/python
# author jiang
# {YEAR}年{MONTH}月{DAY}日
import cv2 as cv
import numpy as np


def KNN():
    train = cv.imread("trainum.png", 0)
    # 24*32
    trainimgs = [train]
    # 腐蚀和膨胀，增强训练集
    for i in range(1, 3):
        kernel = np.ones((i, i), np.uint8)#创建了一个大小为(i, i)的二维数组，用作形态学运算的结构元素
        j = cv.erode(train, kernel)
        trainimgs.append(j)
        r = cv.dilate(train, kernel)
        trainimgs.append(r)
    # 生成knn对象
    knn = cv.ml.KNearest_create()  # 创建knn分类器
    # 训练knn模型，train用于KNN分类器的训练
    for trainimg in trainimgs:
        cells = [np.hsplit(row, 30) for row in np.vsplit(trainimg, 11)]#将图像划分为大小为24×32的小图像

        x = np.array( cells)
        # print(x[1][1])
        trn = x[:, :].reshape(-1, 768).astype(np.float32)#展成一维数组，-1行数自动计算
        k = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        train_label = np.repeat(k, 30)#k重复输入30次
        knn.train(trn, cv.ml.ROW_SAMPLE, train_label) #将训练集和对应的标签转递给KNN分类器对象

    cell = [np.hsplit(row, 30) for row in np.vsplit(train, 11)]#垂直分割成11个子数组，然后对子数组进行水平分割，得到一个嵌套的二维列表cell
    x = np.array(cell)
    # print(x[1][1])
    train = x[:, :].reshape(-1, 768).astype(np.float32)
    t = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    train_label = np.repeat(t, 30)
    return knn, train, train_label


def main():
    knn, train, train_label = KNN()
    #进行不同比重训练集和测试集
    test = train.copy()
    test_label = train_label.copy()
    ret, result, neighbours, dist = knn.findNearest(test, 3)
    right = 0
    for i in range(330):
        if result[i] == test_label[i]:
            right += 1
    print(f'{len(test):}个测试数据识别正确{right:}个')
    # 计算正确率
    ac = right / result.size
    print(f'正确率{ac * 100:.2f}%')


if __name__ == '__main__':
    main()
    cv.waitKey(0)
