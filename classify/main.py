#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
"""
@author: 
@file: run.py
@time: 2020/11/24 15:57
@desc: 
"""
import os
import time

import numpy as np
import pandas as pd
from scipy.ndimage import shift
from sklearn.datasets import fetch_openml
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict
from sklearn.base import clone, BaseEstimator
import matplotlib
import matplotlib.pyplot as plt
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class Never2Classifier(BaseEstimator):
    def fit(self, X, y=None):
        pass

    def predict(self, X):
        return np.zeros((len(X), 1), dtype=bool)


class Classify(object):
    def __init__(self):
        self.MNIST_PATH = "datasets/"
        self.mnist = self.load_mnist_data()

    # 加载数据
    def load_mnist_data(self):
        mnist = fetch_openml('mnist_784')
        mnist.target = mnist.target.astype(np.int8)
        return mnist

    # 生成训练集 测试集
    def split_train_test(self):
        X, y = self.mnist["data"], self.mnist["target"]
        # print(X.shape)
        # print(y.shape)

        # 测试
        some_digit = X[36001]
        # some_digit_image = some_digit.reshape(28, 28)
        # plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
        # plt.axis("off")
        # plt.show()
        # print(y[36001])
        self.some_digit = some_digit

        # 生成训练集 测试集
        X_train, self.X_test, y_train, self.y_test = X[:60000], X[60000:], y[:60000], y[60000:]
        # 打乱训练集 避免有些重复的数字连在一起
        shuffle_index = np.random.permutation(60000)
        self.X_train, self.y_train = X_train[shuffle_index], y_train[shuffle_index]


    def binary_classifier(self):
        self.y_train_2 = (self.y_train == 2)
        # y_test_5 = (y_test==5)

        # 随机梯度下降分类器(SGD)
        # 能够有效处理非常大型的数据集

        # 线性模型
        # 工作原理：为每个像素分配一个每个类别的权重，当他看到新的图像时，将加权后的像素强度汇总，从而得到一个分数进行分类。
        self.sgd_clf = SGDClassifier(random_state=42)
        self.sgd_clf.fit(self.X_train, self.y_train_2)
        # 检测数字2
        # result = self.sgd_clf.predict([self.some_digit])
        # print(result)

    def performance_appraisal(self):
        # 1/交叉验证测量精度
        # 自定义交叉验证函数cross_val_score
        def CrossValScore():
            # cross_val_score预测
            """
            将数据集平均分割成K个等份
            使用1份数据作为测试数据，其余作为训练数据
            计算测试准确率
            使用不同的测试集，重复2、3步骤
            对测试准确率做平均，作为对未知数据预测准确率的估计
            """
            skfolds = StratifiedKFold(n_splits=3, random_state=42, shuffle=True)

            for train_index, test_index in skfolds.split(self.X_train, self.y_train_2):
                clone_clf = clone(self.sgd_clf)
                X_train_folds = self.X_train[train_index]
                y_train_folds = (self.y_train_2[train_index])
                # print(test_index)
                X_test_fold = self.X_train[test_index]
                y_test_fold = (self.y_train_2[test_index])

                clone_clf.fit(X_train_folds, y_train_folds)
                y_pred = clone_clf.predict(X_test_fold)
                # print("y_pred:%s" % len(y_pred))
                n_correct = sum(y_pred == y_test_fold)
                # print(n_correct / len(y_temp))
            # cv=3表示3个折叠 其中2个折叠用来训练 1个用来预测
            # scoring="accuracy" 返回正确率
            result = cross_val_score(self.sgd_clf, self.X_train, self.y_train_2, cv=3, scoring="accuracy")
            print("cross_val_score:%s" % result)

            # 检测模型的准确度 毕竟此模型以2为基准训练
            never_2_clf = Never2Classifier()
            result2 = cross_val_score(never_2_clf, self.X_train, self.y_train_2, cv=3, scoring="accuracy")
            print("模型准确度:%s" % result2)
        CrossValScore()
        # print("y_pred:{}".format(y_pred))

        # 2/混淆矩阵
        def confusion():
            """
            总体思路就是统计A类别实例被分成为B类 别的次数， 例如，要想知道分类器将数字3和数字5混消多少次，只需要通过混消矩阵的第5行第3列来查看
            """
            # 与cross_val score 〈) 国数一样，cross_val_predict〈) 函数同样执行K-fold交叉验证，但返回的不是评价分数，而是每个折叠的预
            # 测。这意味着对于每个实例都可以得到一个干净的预测 〈“干净”的意思是模型预测时使用的数据，在其训练期间从未见过) 。
            y_train_pred = cross_val_predict(self.sgd_clf, self.X_train, self.y_train_2, cv=3)
            result = confusion_matrix(self.y_train_2, y_train_pred)
            print("混淆矩阵：%s" % result)

            """
            [[52349  1693]
                [  711  5247]]
            结果为2x2矩阵
            行为实际类别 列为预测类别
            第一行表示所有非2的图片中 52349张被正确分为非2类别(真负类) 1693张被错误的分为"2"(假正类)
            第二行表示所有"2"的图片中，711张被错误分为非2类别(假负类)  1077张被正确的分为"2"(真正类)
            
            精  度: 真正类 / (真正类 + 假正类)(预测正类) 几个对了几个 和预测值有关
            召回率: 真正类 / (真正类 + 假负类)(实际正类) 几个里找到了几个 和实际值有关
            """
            # 精度
            precision = precision_score(self.y_train_2, y_train_pred)
            print("精度:%s" % precision)
            # 召回率
            recall = recall_score(self.y_train_2, y_train_pred)
            print("召回率:%s" % recall)
            # F1分数：精度和召回率的谐波平均值：会给予较低的评论值更高的权重 因此 只有当召回率和精度都很高时，分类器才能得到较高的F1分数
            # F1 = 2 / (1/精度 + 1/召回率)
            f1score = f1_score(self.y_train_2, y_train_pred)
            print("f1分数:%s" % f1score)

            # 使用decision_function()返回每个实例的分数，然后就可以根据这些分数，使用任意阀值进行预测
            y_scores = self.sgd_clf.decision_function([self.some_digit])
            print("实例分数:%s" % y_scores)  # 8951

            # 阀值 为 0：与predict()方法一样
            threshold = 0
            y_some_digit_pred = (y_scores > threshold)
            print("阀值为0的情况下判断结果:%s" % y_some_digit_pred)

            threshold = 9000
            y_some_digit_pred = (y_scores > threshold)
            print("阀值为7000的情况下判断结果:%s" % y_some_digit_pred)

            # 如何使用适当的阀值，利用cross_val_predict()获取所有实例的分数
            y_scores = cross_val_predict(self.sgd_clf, self.X_train, self.y_train_2, cv=3, method="decision_function")

            # 使用precision_recall_curve()函数计算所有可能的阀值的精度和召回率
            precisions, recalls, thresholds = precision_recall_curve(self.y_train_2, y_scores)
            # print("所有可能的精度:{}, 召回率:{}, 阀值:{}".format(precisions, recalls, thresholds))

            # 使用Matplotlib绘制制度和召回率相对于阀值的函数图
            def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
                # plot参数: 横轴 纵轴 线型:b：blue ；g:green --：虚线; -:实线 label：图例 linewidth:线宽
                plt.plot(thresholds, precisions[:-1], "b--", label="Precision", linewidth=2)
                plt.plot(thresholds, recalls[:-1], "g-", label="Recall", linewidth=2)
                plt.xlabel("Threshold", fontsize=16)
                # legend： loc 图例的位置 fontsize:尺寸
                plt.legend(loc="upper left", fontsize=16)
                # ylim 设置y轴限制
                plt.ylim([0, 1])

            plt.figure(figsize=(8, 4))
            plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
            plt.show()

            # 精度与召回率的比
            def plot_precision_vs_recall(precisions, recalls):
                plt.plot(recalls, precisions, "b-")
                plt.xlabel("Recall", fontsize=16)
                plt.ylabel("Precision", fontsize=16)
                plt.axis([0, 1, 0, 1])

            plt.figure(figsize=(8, 6))
            plot_precision_vs_recall(precisions, recalls)
            # plt.show()
            # 这里的2000确定标准来源于第一张图当精度达到90以上时
            y_train_pred_90 = (y_scores > 2000)
            result = precision_score(self.y_train_2, y_train_pred_90)
            result2 = recall_score(self.y_train_2, y_train_pred_90)
            print("选择阀值为5000下的精度:{}, 召回率:{}".format(result, result2))

            # ROC曲线  召回率/假正类 区别PR曲线 精度/召回率
            fpr, tpr, thresholds = roc_curve(self.y_train_2, y_scores)
            def plot_roc_curve(fpr, tpr, label=None):
                plt.plot(fpr, tpr, linewidth=2, label=label)
                plt.plot([0, 1], [0, 1], 'k--')
                plt.axis([0, 1, 0, 1])
                plt.xlabel(["False Positive Rate"], fontsize=16)
                plt.ylabel(["True Positive Rate"], fontsize=16)

            plt.figure(figsize=(8, 6))
            plot_roc_curve(fpr, tpr)
            # plt.show()

            # 比较分类器:测量曲线线下面积(AUC) 完美的分类起ROC的AUC=1 纯随机分类器的AUC为0.5就是虚线下方的面积
            # roc_auc_score()
            ans = roc_auc_score(self.y_train_2, y_scores)
            print("分类器的AUC:%s" % ans)

            # 训练一个随机森林分类器与当前的二元分类器的auc比较一下
            forest_clf = RandomForestClassifier(random_state=42)
            y_probas_forest = cross_val_predict(forest_clf, self.X_train, self.y_train_2, cv=3, method="predict_proba")
            y_scores_forest = y_probas_forest[:, 1]
            fpr_forest, tpr_forest, thresholds_forest = roc_curve(self.y_train_2, y_scores_forest)

            plt.plot(fpr, tpr, "b:", label="SGD")
            plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
            plt.legend(loc="lower right")
            # plt.show()

            # 测试随机森林的auc分数与精度和召回率
            auc_forest = roc_auc_score(self.y_train_2, y_scores_forest)
            print("随机森林的ACU:%s" % auc_forest)
            y_train_pred_forest = cross_val_predict(forest_clf, self.X_train, self.y_train_2, cv=3)
            precision_forest = precision_score(self.y_train_2, y_train_pred_forest)
            recall_forest = recall_score(self.y_train_2, y_train_pred_forest)
            print("随机森林的精度:%s, 召回率:%s" % (precision_forest, recall_forest))

        confusion()

    def multi_class_classifier(self):
        # OvA一对多策略 数字分为10类 则创建10个二元分类器 分别给出自己的分数 高者为胜
        self.sgd_clf.fit(self.X_train, self.y_train)
        some_digit_scores = self.sgd_clf.decision_function([self.some_digit])
        print("10个类别的分数:%s" % some_digit_scores)
        # 输出得分最高的数字
        num = np.argmax(some_digit_scores)
        print("得分最高的数字:%s" % num)
        # 目标类别的列表
        print("目标类别的列表:%s" % self.sgd_clf.classes_)

        # OvO强制转化为一对一策略分类器 为每一对数字训练一个二元分类器 本例则需要45个二元分类器 OneVsRestClassifier
        ovo_clf = OneVsRestClassifier(SGDClassifier(random_state=42))
        ovo_clf.fit(self.X_train, self.y_train)
        result = ovo_clf.predict([self.some_digit])
        print("OvO预测2的结果:%s" % result)
        print("OvO分类器个数:%s" % len(ovo_clf.estimators_))

        # 随机森林分类器直接就可以将实例分为多个类别 调用predict_proba()可以获得分类器将每个实例分类为每个类别的概率列表
        forest_clf = RandomForestClassifier(random_state=42)
        forest_clf.fit(self.X_train, self.y_train)
        result = forest_clf.predict([self.some_digit])
        print("随机森林预测结果:%s" % result)
        result2 = forest_clf.predict_proba([self.some_digit])
        print("随机森林获取每个实例的概率列表:%s" % result2)

        # 采用交叉验证评估分类器
        result3 = cross_val_score(self.sgd_clf, self.X_train, self.y_train, cv=3, scoring="accuracy")
        print("交叉验证评估分数:%s" % result3)
        # 输入简单缩放提高准确率
        scaler = StandardScaler()
        self.X_train_scaled = scaler.fit_transform(self.X_train.astype(np.float64))
        result4 = cross_val_score(self.sgd_clf, self.X_train_scaled, self.y_train, cv=3, scoring="accuracy")
        print("简单错放后的交叉验证分数:%s" % result4)

    def error_analysis(self):
        y_train_pred = cross_val_predict(self.sgd_clf, self.X_train_scaled, self.y_train, cv=3)
        conf_mx = confusion_matrix(self.y_train, y_train_pred)
        # matshow 将矩阵或者数组绘制成图像的函数
        plt.matshow(conf_mx, cmap=plt.cm.gray)
        plt.show()

        # 获取混淆矩阵中错误率
        # axis=1 行统计 keedims=True 结果为一维矩阵
        row_sums = conf_mx.sum(axis=1, keepdims=True)
        norm_conf_mx = conf_mx / row_sums
        # 0填充对角线 对角线为正确的预测 则 对角线为全黑
        # fill_diagonal 填充对角
        np.fill_diagonal(norm_conf_mx, 0)
        plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
        plt.show()

    def multiple_label_classification(self):
        # 分类器为每个实例产出多个类别 比如一张照片识别出多张人脸 亦或者 一个数字同时区分她是否为奇数与是否<7
        y_train_large = (self.y_train >= 7)
        y_train_even = (self.y_train % 2 == 0)
        y_multilabel = np.c_[y_train_large, y_train_even]

        knn_clf = KNeighborsClassifier()
        self.knn_clf = knn_clf
        knn_clf.fit(self.X_train, y_multilabel)
        result = knn_clf.predict([self.some_digit])
        print("预测是否大于6与是否为偶数:%s" % result)

        # F1分数评估多标签分类器
        y_train_knn_pred = cross_val_predict(knn_clf, self.X_train, y_multilabel, cv=3, n_jobs=1)
        # average = "macro"
        """
        macro：不考虑类别数量，不适用于类别不均衡的数据集，其计算方式为： 各类别的精确率求和/类别数量
        weighted:各类别的精确率 × 该类别的样本数量（实际值而非预测值）/ 样本总数量
        
        对于类别不均衡的分类模型，采用macro方式会有较大的偏差，
        采用weighted方式则可较好反映模型的优劣，
        因为若类别数量较小则存在蒙对或蒙错的概率，其结果不能真实反映模型优劣，
        需要较大的样本数量才可计算较为准确的评价值，通过将样本数量作为权重，可理解为评价值的置信度，数量越多，其评价值越可信
        """
        knn_f1 = f1_score(self.y_train, y_train_knn_pred, average="macro")
        print("多标签分类器的f1分数:{}".format(knn_f1))

    def plot_digit(self, data):
        image = data.reshape(28, 28)
        plt.imshow(image, cmap=matplotlib.cm.binary,
                   interpolation="nearest")
        plt.axis("off")


    def multiple_output_classification(self):
        # 多输出-多类别分类：多标签分类的泛化 标签也可以是多种类别的(他可以有两个以上可能的值)
        """
        numpy.random.randint(low, high=None, size=None)
        返回一个随机整型数，范围从低（包括）到高（不包括）
        """
        noise = np.random.randint(0, 100, (len(self.X_train), 784))
        X_train_mod = self.X_train + noise
        noise = np.random.randint(0, 100, (len(self.X_test), 784))
        X_test_mod = self.X_test + noise

        y_train_mod = self.X_train
        y_test_mod = self.X_test

        some_index = 5500
        # subplot:在一张图展示多个子图 121:表示将原始图像切割为1行2列 并定位到第一张图
        plt.subplot(121);self.plot_digit(X_test_mod[some_index])
        plt.subplot(122);self.plot_digit(y_test_mod[some_index])
        plt.show()

        knn_clf = KNeighborsClassifier()
        knn_clf.fit(X_train_mod, y_train_mod)
        clean_digit = knn_clf.predict([X_test_mod[some_index]])
        self.plot_digit(clean_digit)

    def exercise(self):
        def first_exercise():
            from sklearn.model_selection import GridSearchCV

            # weights参数默认是uniform，该参数将所有数据看成等同的，
            # 而另一值是distance，它将近邻的数据赋予更高的权重，而较远的数据赋予较低权重

            param_grid = [
                {"weights": ["uniform", "distance"], "n_neighbors":[3, 4, 5]}
            ]

            knn_clf = KNeighborsClassifier()
            # n_jobs 设定工作的core数量 等于-1的时候 表示cpu里的所有core进行工作
            # verbose 日志冗余程度 0 不输出 1 偶尔输出 >1 每个子模型都输出
            grid_search = GridSearchCV(knn_clf, param_grid, cv=5, verbose=3, n_jobs=-1)
            grid_search.fit(self.X_train, self.y_train)
            print("最优超参数组合:{}".format(grid_search.best_params_))
            print("最优超参数组合分数:{}".format(grid_search.best_score_))

            y_pred = grid_search.predict(self.X_test)
            score = accuracy_score(self.y_test, y_pred)
            print("最优超参数组合在测试集的精度:{}".format(score))
            return grid_search
        # grid_search = first_exercise()

        def second_exercise():
            def shift_image(image, dx, dy):
                image = image.reshape((28, 28))
                # shift(image, [1,2]) 将图片image向下移动1个像素并向右移动2个元素
                shifted_image = shift(image, [dy, dx], cval=0)
                return shifted_image.reshape([-1])

            # image = self.X_train[1000]
            # shifted_image_down = shift_image(image, 0, 5)
            # shifted_image_left = shift_image(image, -5, 0)

            # plt.figure(figsize=(12, 3))
            # plt.subplot(131)
            # plt.title("Original", fontsize=14)
            # plt.imshow(image.reshape(28, 28), interpolation="nearest", cmap=matplotlib.cm.gray)
            # plt.subplot(132)
            # plt.title("shifted down", fontsize=14)
            # plt.imshow(shifted_image_down.reshape(28, 28), interpolation="nearest", cmap=matplotlib.cm.gray)
            # plt.subplot(133)
            # plt.title("shifted left", fontsize=14)
            # plt.imshow(shifted_image_left.reshape(28, 28), interpolation="nearest", cmap=matplotlib.cm.gray)
            # plt.show()

            X_train_augmented = [image for image in self.X_train]
            y_train_augmented = [label for label in self.y_train]

            for dx, dy in [[-1,0], [1, 0], [0, 1], [0, -1]]:
                for image, label in zip(self.X_train, self.y_train):
                    shifted_image = shift_image(image, dx, dy)
                    X_train_augmented.append(shifted_image)
                    y_train_augmented.append(label)
            X_train_augmented = np.array(X_train_augmented)
            y_train_augmented = np.array(y_train_augmented)

            shuffle_idx = np.random.permutation(len(X_train_augmented))
            X_train_augmented = X_train_augmented[shuffle_idx]
            y_train_augmented = y_train_augmented[shuffle_idx]
            knn_clf = KNeighborsClassifier(**{"n_neighbors": 4, "weights": "distance"})
            knn_clf.fit(X_train_augmented, y_train_augmented)
            y_pred = knn_clf.predict(self.X_test)

            score = accuracy_score(self.y_test, y_pred)
            print("数据增广后的k-近邻分类器的分数:{}".format(score))

        second_exercise()

    def run(self):
        # 生成训练集与测试集
        self.split_train_test()
        time1 = time.time()
        print("------生成训练集与测试集花费时间:%s-------" % (time1 - begin))

        # 训练一个二元分类器
        # self.binary_classifier()
        # time2 = time.time()
        # print("------训练一个二元分类器花费时间:%s-------" % (time2 - time1))

        # 性能考核
        # self.performance_appraisal()
        # time3 = time.time()
        # print("------性能考核花费时间:%s-------" % (time3 - time2))

        # 多类别分类器
        # self.multi_class_classifier()
        # time4 = time.time()
        # print("------多类别分类器花费时间:%s-------" % (time4 - time3))

        # 错误分析
        # self.error_analysis()
        # time5 = time.time()
        # print("------错误分析花费时间:%s-------" % (time5 - time4))

        # 多标签分类
        # self.multiple_label_classification()
        # time6 = time.time()
        # print("------多标签分类花费时间:%s-------" % (time6 - time1))

        # 多输出分类
        # self.multiple_output_classification()
        # time7 = time.time()
        # print("------多输出分类花费时间:%s-------" % (time7 - time1))

        # 练习
        self.exercise()

begin = time.time()
Classify().run()
end = time.time()
print('---time:{}'.format(end - begin))
# Classify().test()
