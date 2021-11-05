#随机森林算法
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
#下面我们直接使用数据集自带的训练集和测试集来训练模型和测试模型
#the train.csv and test.csv are loading into the process
train=pd.read_csv('data/train.csv')
test=pd.read_csv('data/test.csv')
print(train.head())
#删除数据集中的缺失值
#train=train.dropna(axis=0)

#pull data into target (y) and predictors (X)
train_y=train.SalePrice
#挑选部分列
predictor_cols=['LotArea','OverallQual','YearBuilt','1stFlrSF','2ndFlrSF','FullBath','BedroomAbvGr','TotRmsAbvGrd']
#直方图
#y_train = train.pop('SalePrice')
#y_train.hist()
#y_train = np.log1p(y_train)
# 相关图
corrmat = train.corr()

plt.subplots(figsize=(12,9))
sns.heatmap(corrmat, vmax=0.9, square=True)
plt.show()
#缺失率
df = pd.concat((train, test), axis=0)
print(pd.isnull(df).sum())
print(df.isnull().sum().sort_values(ascending=False).head(10))
df_na = (df.isnull().sum() / len(df)) * 100
df_na = df_na.drop(df_na[df_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'缺失率': df_na})
print(missing_data.head(10))
f, ax = plt.subplots(figsize=(15,12))
plt.xticks(rotation='90')
sns.barplot(x=df_na.index, y=df_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)
plt.show()
#影响价格的前十个变量

import matplotlib.pyplot as plt
import seaborn as sns
k  = 10 # 关系矩阵中将显示10个特征
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index ##显示和saleprice相近的十个关系变量矩阵
cm = np.corrcoef(train[cols].values.T)
sns.set(font_scale=1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
plt.show()
#Create training predictors data
train_X=train[predictor_cols]

my_model=RandomForestRegressor()
#训练随机森林模型
my_model.fit(train_X,train_y)
'''
RandomForestRegressor(bootstrap=True, criterion='mse', max_depth=None,
           max_features='auto', max_leaf_nodes=None,
           min_impurity_decrease=0.0,
           min_samples_leaf=1, min_samples_split=2,
           min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
           oob_score=False, random_state=None, verbose=0, warm_start=False)
#Read the test data 加载测试集数据
'''

#Treat the test data in the same way as training data. In this case,pull same columns.
#这里注意要和训练集挑选的列相同
test_X=test[predictor_cols]
#Use the model to make predictions
#对测试集进行预测
predicted_prices=my_model.predict(test_X)
#We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)
#新建DataFrame保存预测结果，分别是Id和预测对结果
#my_submission=pd.DataFrame({'Id':test.Id,'SalePrice':predicted_prices})

#保存预测结果到新的csv文件中
#my_submission.to_csv('data/submission.csv',index=False)
# 相关图
from sklearn.metrics import mean_absolute_error
#决策树模型的参数max_leaf_nodes会影响模型表现，自定义函数探究该变量
def get_mae(train_X, val_X,train_y,val_y):
    model = RandomForestRegressor()
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)
#在实际应用中，我们应该将原始数据集划分为训练集和测试集
#训练集用来训练模型，测试集用来测试模型表现
from sklearn.model_selection import train_test_split
# split data into training and validation data, for both predictors and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(train_X, train_y,random_state = 0)


my_mae = get_mae(train_X, val_X, train_y, val_y)
print("mae:  %d" %(my_mae))