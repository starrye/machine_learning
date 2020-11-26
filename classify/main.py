#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
"""
@author: 
@file: main.py
@time: 2020/11/24 15:57
@desc: 
"""
import os
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
import matplotlib
import matplotlib.pyplot as plt


class Classify(object):
    def __init__(self):
        self.MNIST_PATH = "datasets/"
        self.mnist = self.load_mnist_data()

    # 加载数据
    def load_mnist_data(self):
        mnist = fetch_openml('mnist_784')
        return mnist

    # 生成训练集 测试集
    def split_train_test(self):
        X, y = self.mnist["data"], self.mnist["target"]
        #print(X.shape)
        #print(y.shape)


        # 测试
        some_digit = X[36000]
        some_digit_image = some_digit.reshape(28, 28)
        plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
        plt.axis("off")
        plt.show()
        print(y[36000])
        self.some_digit = some_digit

        # 生成训练集 测试集
        X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

        # 打乱训练集 避免有些重复的数字连在一起
        import numpy as np
        shuffle_index = np.random.permutation(60000)
        self.X_train, self.y_train = X_train[shuffle_index], y_train[shuffle_index]

    def binary_classifier(self):
        y_train_9 = (self.y_train == "9")
        #y_test_5 = (y_test==5)

        # 随机梯度下降分类器(SGD)
        # 能够有效处理非常大型的数据集
        from sklearn.linear_model import SGDClassifier
        sgd_clf = SGDClassifier(random_state=42)
        sgd_clf.fit(self.X_train, y_train_9)
        # 检测数字9
        result = sgd_clf.predict([self.some_digit])
        print(result)


    def run(self):
        # 生成训练集与测试集
        self.split_train_test()

        # 训练一个二元分类器
        self.binary_classifier()






Classify().run()
