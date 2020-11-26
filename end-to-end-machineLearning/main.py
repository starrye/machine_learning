#!/usr/local/bin/python3
# -*- coding:utf-8 -*-
"""
@author: 
@file: main.py
@time: 2020/11/5 10:12
@desc: 
"""
from typing import List

import pandas as pd
import os
import matplotlib.pyplot as plt
import numpy as np
import hashlib
# 划分测试集
from sklearn.model_selection import train_test_split
# 分层抽样
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, LabelBinarizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor

# 转换器
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
        self.rooms_ix = 3
        self.bedrooms_ix = 4
        self.population_ix = 5
        self.household_ix = 6

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        # 每栋房子平均有多少房间
        rooms_per_household = X[:, self.rooms_ix] / X[:, self.household_ix]
        # 每栋房子平均有多少人
        population_per_household = X[:, self.population_ix] / X[:, self.household_ix]
        if self.add_bedrooms_per_room:
            # 卧室占比
            bedrooms_per_room = X[:, self.bedrooms_ix] / X[:, self.rooms_ix]
            # np.c_是按行连接两个矩阵，就是把两矩阵左右相加，要求行数相等，类似于pandas中的merge()。
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


class DataFrameSelector(BaseEstimator, TransformerMixin):
    """
    scikit-learn中没有可以处理pandas dataframe的 所以需要实现一个简单的自定义转换器
    """
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        return X[self.attribute_names].values


class PipelineFriendlyLabelBinarizer(LabelBinarizer):
    def fit_transform(self, X, y=None):
        return super(PipelineFriendlyLabelBinarizer, self).fit_transform(X)


class RealEstateInvestment(object):
    def __init__(self):
        self.HOUSING_PATH = "end-to-end-machineLearning/datasets/housing"
        self.housing = self.load_housing_data()

    # 加载数据
    def load_housing_data(self):
        csv_path = os.path.join(self.HOUSING_PATH, "housing.csv")
        return pd.read_csv(csv_path)

    # 浏览数据
    def view_the_data(self):
        # 打印csv数据前5行
        # print(self.housing.head())

        # 快速获取数据机的简单描述，包括行数，类型，非空值的数量 内存占用
        # self.housing.info()

        # 获取有多少种分类与隶属该分类下区域数量
        # print(self.housing["ocean_proximity"].value_counts())

        # 显示数值属性的摘要
        # std:标准差(测量数值的离散程度), 25%: 有25%的区域低于当前数值 50%, 75%类似
        # print(self.housing.describe())

        # 绘制直方图
        # bins 条状图的条数
        # figsize: 图形尺寸
        self.housing.hist(bins=50, figsize=(20, 15))
        plt.show()

    # 创建测试集的几种方案
    def split_train_test(self,test_ratio):
        # 弊端：随机测试集每次都不一样 避免
        shuffled_indices = np.random.permutation(len(self.housing))  # 生成一个随机序列 范围是数据集量
        test_set_size = int(len(self.housing) * test_ratio)  # 测试集数量:总数 x 占比test_ratio
        test_indices = shuffled_indices[:test_set_size]
        train_indices = shuffled_indices[test_set_size:]
        return self.housing.iloc[train_indices], self.housing.iloc[test_indices]  # 切割数据集 返回测试集与训练集

    # 使用标识符选取确定的测试集 hashlib
    # 计算每个实例标识符的hash值 只取hash的最后一个字节，如果该值<=51 (256的20%) 则选入测试集
    def test_set_check(self, identifier, test_ratio, hash):
        return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio

    def split_train_test_by_id(self, data, test_ratio, id_column, hash=hashlib.md5):
        ids = data[id_column]
        in_test_set = ids.apply(lambda id_: self.test_set_check(id_, test_ratio, hash))
        return data.loc[-in_test_set], data.loc[in_test_set]

    def create_income_type(self):
        # ceil:向上取整函数 floor:向下取整函数
        self.housing["income_cat"] = np.ceil(self.housing["median_income"] / 1.5)
        # Series.where(cond, other=nan, inplace=False, axis=None, level=None, errors=‘raise’, try_cast=False, raise_on_error=None)
        # 如果 cond 为真，保持原来的值，否则替换为other， inplace为真标识在原数据上操作，为False标识在原数据的copy上操作
        # 将大于5的类别合并为类别5
        self.housing["income_cat"].where(self.housing["income_cat"] < 5, 5.0, inplace=True)

        split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        for train_index, test_index, in split.split(self.housing, self.housing["income_cat"]):
            self.strat_train_set = self.housing.loc[train_index]
            self.strat_test_set = self.housing.loc[test_index]
        # print(self.housing["income_cat"].value_counts() / len(self.housing))
        for set in (self.strat_train_set, self.strat_test_set):
            set.drop(["income_cat"], axis=1, inplace=True)


    def test_data(self):
        # 创建测试集
        # 1.随机序列然后切割
        # train_set, test_set = data.split_train_test(0.2)
        # print(len(train_set), "train +", len(test_set), "test")
        # 2.行索引
        # 如果只是在数据集的末尾添加数据 而不会删除任何行 可以拿行索引作为唯一标识符
        # reset_index()重置索引 当删除某些行 或出现索引不连续问题 使用reset可以重置索引 旧的索引会成为新的数据列 叫index 如果不想保留旧的索引 使用参数drop=True
        # housing_with_id = self.housing.reset_index()
        # train_set, test_set = self.split_train_test_by_id(housing_with_id, 0.2, "index")
        # 3.稳定的特征为标识符
        # 如果担心会删除行索引 可以尝试使用某个最稳定的特征来创建唯一标识符，例如经纬度
        # housing_with_id["id"] = self.housing["longitude"] * 1000 + self.housing["latitude"]
        # train_set, test_set = self.split_train_test_by_id(housing_with_id, 0.2, "id")
        # 4.利用内置函数 train_test_split
        # test_size:测试集占比
        # random_state随机数种子：其实就是该组随机数的编号，在需要重复试验的时候，保证得到一组一样的随机数。比如你每次都填1，其他参数一样的情况下你得到的随机数组是一样的。但填0或不填，每次都会不一样
        # train_set, test_set = train_test_split(self.housing, test_size=0.2, random_state=42)
        # print(len(train_set))
        # print(len(test_set))
        self.create_income_type()
        pass

    def data_visualization(self):
        # 地理位置可视化
        self.housing = self.strat_train_set.copy()
        self.housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4, s=self.housing["population"] / 100, label="population", c="median_house_value",
                          cmap=plt.get_cmap("jet"), colorbar=True)
        plt.legend()
        """
        plot参数:
           s:各个点的大小，数值越大，对应点越大
           c:为每一点赋予颜色 
           cmap:选择色盘，jet属于蓝红区间色盘
           colorbar:颜色区间展示 类似图例
           label	用于图例的标签
           ax	要在其上进行绘制的matplotlib subplot对象。如果没有设置，则使用当前matplotlib subplot
           style	将要传给matplotlib的风格字符串(for example: ‘ko–’)
           alpha	图表的填充不透明(0-1)
           kind	可以是’line’, ‘bar’, ‘barh’, ‘kde’
           logy	在Y轴上使用对数标尺
           use_index	将对象的索引用作刻度标签
           rot	旋转刻度标签(0-360)
           xticks	用作X轴刻度的值
           yticks	用作Y轴刻度的值
           xlim	X轴的界限
           ylim	Y轴的界限
           grid	显示轴网格线

       kind : str
           ‘line’ : line plot (default)#折线图
           ‘bar’ : vertical bar plot#条形图
           ‘barh’ : horizontal bar plot#横向条形图
           ‘hist’ : histogram#柱状图
           ‘box’ : boxplot#箱线图
           ‘kde’ : Kernel Density Estimation plot#Kernel 的密度估计图，主要对柱状图添加Kernel 概率密度线
           ‘density’ : same as ‘kde’
           ‘area’ : area plot#不了解此图
           ‘pie’ : pie plot#饼图
           ‘scatter’ : scatter plot#散点图  需要传入columns方向的索引
       """
        plt.show()

    def data_relation(self):
        # 寻找相关性
        # 1.corr：计算每个参数与其他参数的相关性 与自己相关性为1
        # corr_matrix = self.housing.corr()
        # print(corr_matrix["median_house_value"].sort_values(ascending=False))
        # 相关系数从-1到1 越接近1越表现出越强的正相关 仅测量线性相关性

        # 2.scatter_matrix, 绘制出每个数值属性相对于其他数值属性的相关性
        attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
        scatter_matrix(self.housing[attributes], figsize=(12, 8))
        """
        scatter_matrix(frame, alpha=0.5, figsize=None....)
        1。frame，pandas dataframe对象
        2。alpha， 图像透明度，一般取(0,1]
        3。figsize，以英寸为单位的图像大小，一般以元组 (width, height) 形式设置
        """
        # plt.show()

        self.housing.plot(kind="scatter", x="median_income", y="median_house_value", c="median_house_value", alpha=0.1,cmap=plt.get_cmap("jet"), colorbar=True)
        plt.show()

    def data_explore(self):
        # 数据可视化
        self.data_visualization()
        # 寻找相关性
        self.data_relation()

    def data_clean(self):
        """
        面对缺失值三种处理方式：
            1.去除含有缺失值的样本(行)
            2.将含有缺失值的列去掉
            3.将缺失值用某些值填充(0，平均值，中值等)
            4.scikit-learn self.imputer函数
        :return:
        """
        # drop：(["列名"]， axis=1)返回删除指定行或者列 axis=0 为行 =1 为列
        # 将预测器和标签分开 因为不一定会采取同样的转换方式
        self.housing = self.strat_train_set.drop("median_house_value", axis=1)
        self.housing_labels = self.strat_train_set["median_house_value"].copy()

        # dropna 清除缺失值：subset:在哪些列中查看是否有缺失值
        # self.housing = self.housing.dropna(subset=["total_bedrooms"])

        # drop 清除这个属性
        # self.housing = self.housing.drop("total_bedrooms", axis=1)

        # median() 获取某一个属性的中位数
        # fillna(某一列填充某个值)
        # median = self.housing["total_bedrooms"].median()
        # self.housing["total_bedrooms"] = self.housing["total_bedrooms"].fillna(median)
        # print(self.housing.info())
        """
        
        strategy:也就是你采取什么样的策略去填充空值，总共有4种选择。
        分别是mean,median, most_frequent,以及constant，这是对于每一列来说的，
        如果是mean，则该列则由该列的均值填充。而median,则是中位数，most_frequent则是众数。
        需要注意的是，如果是constant,则可以将空值填充为自定义的值，这就要涉及到后面一个参数了，也就是fill_value。如果strategy='constant',则填充fill_value的值。
        """
        self.imputer = SimpleImputer(strategy="median")
        self.housing_num = self.housing.drop("ocean_proximity", axis=1)
        # Fit the self.imputer on self.housing_num
        self.imputer.fit(self.housing_num)
        # print(self.imputer.statistics_)
        # print(self.housing_num.median().values)
        X = self.imputer.transform(self.housing_num)
        housing_tr = pd.DataFrame(X, columns=self.housing_num.columns)
        # print(housing_tr.info())

    def text_process(self):
        # encoder = LabelEncoder()
        housing_cat = self.housing["ocean_proximity"]
        # # fit_transform: 填充数据后转换
        """
        文本转整数
        """
        # housing_cat_encoded = encoder.fit_transform(housing_cat)
        # print(housing_cat_encoded)
        # print(encoder.classes_)

        """
        独热编码：机器学习算法会以为两个相近的数字比两个离得比较远的数字更为相似一些，事实并非如此，解决方案：为每个类别创建一个二进制的属性
        当类别是x 一个属性为1 其他为0 当类别是y 另一个属性为1 其他为0。。。。
        因为只有一个为1 其他为0 所以被称为独热编码
        OneHotEncoder
        """
        # encoder = OneHotEncoder()
        # # reshape 允许一个参数为-1 表示未指定：该值由数组的长度和其余纬度来推断
        # housing_cat_1hot = encoder.fit_transform(housing_cat_encoded.reshape(-1, 1))
        # # 结果得到一个scipy稀疏矩阵 仅存储非0元素的位置 因为0太多了
        # print(housing_cat_1hot)
        # # toarray仍可以转化为numpy数组
        # print(housing_cat_1hot.toarray())

        """
        LabelBinarizer可以一次性从文本转为整数再转为独特向量 是上述两个方法的综合
        但是他返回的是 一个 numpy矩阵 通过实例化类的时候传入初始化参数sparse_output=True可以得到稀疏矩阵
        """
        self.encoder = LabelBinarizer(sparse_output=True)
        housing_cat_1hot = self.encoder.fit_transform(housing_cat)
        #print(housing_cat_1hot)

        # 自定义转换器
        attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
        housing_extra_attribs = attr_adder.transform(self.housing.values)
        
        # 转换流水线
        print(self.housing_num)
        self.num_attribs = list(self.housing_num)
        cat_attribs = ["ocean_proximity"]
        num_pipeline = Pipeline([
            ("selector", DataFrameSelector(self.num_attribs)),
            ("imputer", SimpleImputer(strategy="median")),
            ("attribs_adder", CombinedAttributesAdder()),
            ("std_scaler", StandardScaler()),
        ])

        cat_pipeline = Pipeline([
            ("selector", DataFrameSelector(cat_attribs)),
            ("label_binarizer", PipelineFriendlyLabelBinarizer()),
        ])

        self.full_pipeline = FeatureUnion(transformer_list=[
            ("num_pipeline", num_pipeline),
            ("cat_pipeline", cat_pipeline)
        ])
        self.housing_prepared = self.full_pipeline.fit_transform(self.housing)
        print(self.housing_prepared.shape)

    def select_train_models(self):
        # 培训和评估训练集
        # 线性回归模型
        lin_reg = LinearRegression()
        # 训练模型
        lin_reg.fit(self.housing_prepared, self.housing_labels)
        # iloc: 通过行号取数据 这里取前五行的数据
        some_data = self.housing.iloc[:5]
        some_lables = self.housing_labels.iloc[:5]
        some_data_prepared = self.full_pipeline.transform(some_data)
        # predict:对数据进行预测结果
        # print("Predictions:\t", lin_reg.predict(some_data_prepared))
        # 现实结果
        # print("Lables:\t\t", list(some_lables))

        from sklearn.metrics import mean_squared_error
        housing_predictions = lin_reg.predict(self.housing_prepared)
        # mean_squared_error 获取均方误差
        lin_mse = mean_squared_error(self.housing_labels, housing_predictions)
        # 获取均方根误差 (tip:均方根误差表现的是与真实数据之间的偏差，方差是与平均值之间的偏差，方差表现的是数据的离散程度)
        lin_rmse = np.sqrt(lin_mse)
        # print("lin_rmse", lin_rmse)

        # 非线性模型-决策树
        from sklearn.tree import DecisionTreeRegressor
        tree_reg = DecisionTreeRegressor()
        tree_reg.fit(self.housing_prepared, self.housing_labels)
        housing_predictions2 = tree_reg.predict(self.housing_prepared)
        tree_mse = mean_squared_error(self.housing_labels, housing_predictions2)
        tree_rmse = np.sqrt(tree_mse)
        # print("tree_rmse", tree_rmse)

        # 使用交叉验证来更好地进行评估
        from sklearn.model_selection import cross_val_score
        # k-折交叉验证 将训练集随机分为10个子集 每个子集称为一个折叠，然后对决策树模型进行10次训练和评估-每次挑选1个折叠评估，另外9个用来
        # 训练，产出的结果是一个包含10次评估分数的数组
        scores = cross_val_score(tree_reg, self.housing_prepared, self.housing_labels, scoring="neg_mean_squared_error", cv=10)
        rmse_scores = np.sqrt(-scores)
        def display_scores(scores):
            print("Scores:", scores)
            # 均值
            print("Mean:", scores.mean())
            # 标准差(上下浮动)
            print("Standard deviation:", scores.std())
            print('---------------------------')
        # 交叉验证的评分结果
        display_scores(rmse_scores)

        # 线性回归的评分结果
        lin_scores = cross_val_score(lin_reg, self.housing_prepared, self.housing_labels, scoring="neg_mean_squared_error", cv=10)
        lin_rmse_scores = np.sqrt(-lin_scores)
        display_scores(lin_rmse_scores)

        # 随机森林的评分结果
        from sklearn.ensemble import RandomForestRegressor
        forest_reg = RandomForestRegressor()
        forest_reg.fit(self.housing_prepared, self.housing_labels)
        housing_predictions3 = forest_reg.predict(self.housing_prepared)
        forest_mse = mean_squared_error(self.housing_labels, housing_predictions3)
        forest_rmse = np.sqrt(forest_mse)
        forest_scores = cross_val_score(forest_reg, self.housing_prepared, self.housing_labels,
                                     scoring="neg_mean_squared_error", cv=10)
        forest_rmse_scores = np.sqrt(-forest_scores)
        display_scores(forest_rmse_scores)

        # 保存各个模型
        from sklearn.externals import joblib
        # 保存线性回归模型
        joblib.dump(lin_reg, "lin_reg.pk1")
        # 保存决策树模型
        joblib.dump(tree_reg, "tree_reg.pk1")
        # 保存随机森林模型
        joblib.dump(forest_reg, "forest_reg.pk1")

    def fine_tuning_models(self):
        # 网格搜索 -超参数 需要尝试的值 使用交叉验证来评估超参数的所有可能的组合 搜索范围较小时适用
        from sklearn.model_selection import GridSearchCV

        param_grid = [
            {"n_estimators": [3, 10, 30], "max_features": [2, 4, 6, 8]},
            {"bootstrap": [False], "n_estimators": [3, 10], "max_features":[2, 3, 4]},
        ]
        forest_reg = RandomForestRegressor()
        # GridSearchCV:它存在的意义就是自动调参，只要把参数输进去，就能给出最优化的结果和参数
        # 1/sklearn会评估出n_estimators * max_features的组合方式
        # 2/在bootstrap=false的情况下 评估出n_estimators * max_features 的组合方式 最终有 12+6 = 18中不同的超参数组合方式，
        # 3/每一种组合方式要在训练集上训练5次 所以一共训练18x5=90次
        # 4/当训练结束后 可以通过best_params_获得最好的组合方式
        # 5/best_estimatos_获得最好的估算器
        # 6/cv_results_获得评估分数

        # 网格搜索适用于三四个（或者更少）的超参数（当超参数的数量增长时，网格搜索的计算复杂度会呈现指数增长，这时候则使用随机搜索）
        # param_grid：值为字典或者列表，即需要最优化的参数的取值
        # cv: 交叉验证参数
        # scoring :准确度评价标准
        grid_search = GridSearchCV(forest_reg, param_grid, cv=5, scoring="neg_mean_squared_error")
        grid_search.fit(self.housing_prepared, self.housing_labels)
        # print(grid_search.best_params_)
        #print(grid_search.best_estimator_)
        cvres = grid_search.cv_results_
        for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
           print(np.sqrt(-mean_score), params)
        # 根据结果知道 最佳解决方案为将超参数max_features设置为6 n_estimators设置为30 得到方差分数最低

        # 随机搜索 搜索范围较大适用 RandomizeSearchCV
        # 集成方法 表现最有的模型组合起来

        # 分析最佳模型及其错误
        # feature_importtances: 指出每个属性的重要性
        feature_importtances = grid_search.best_estimator_.feature_importances_
        # print(feature_importtances)

        #
        extra_arrtibs = ["room_per_hhold", "pop_per_hhlod", "bedrooms_per_room"]
        cat_one_hot_attribs = list(self.encoder.classes_)
        attributes = self.num_attribs + extra_arrtibs + cat_one_hot_attribs
        result = sorted(zip(feature_importtances, attributes), reverse=True)
        # print(result)

        # 通过测试集评估系统
        final_model = grid_search.best_estimator_
        X_test = self.strat_test_set.drop("median_house_value", axis=1)
        Y_test = self.strat_test_set["median_house_value"].copy()

        X_test_prepared = self.full_pipeline.transform(X_test)
        final_predictions = final_model.predict(X_test_prepared)

        final_mse = mean_squared_error(Y_test, final_predictions)
        final_rmse = np.sqrt(final_mse)
        print(final_rmse)

        scores = cross_val_score(final_model, self.housing_prepared, self.housing_labels, scoring="neg_mean_squared_error",
                                 cv=10)
        final_rmse_scores = np.sqrt(-scores)
        print(final_rmse_scores)



    def run(self):
        # 测试集
        self.test_data()

        # 数据探索和可视化
        self.data_explore()

        # 数据清理
        self.data_clean()

        # 处理文本和分类属性
        self.text_process()

        # 选择和训练模型
        # self.select_train_models()

        # 微调模型
        self.fine_tuning_models()

data = RealEstateInvestment()
data.run()
