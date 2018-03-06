import pandas as pd
from sklearn import linear_model
import seaborn
import matplotlib.pyplot as plt
from sklearn.model_selection import cross_val_score as cvs
import numpy

iris = pd.read_csv('iris.csv')

#散点图观察petal_length 和 petal_width关系
seaborn.regplot(x='petal_length',y='petal_width',data=iris)
plt.show()

#训练线性回归模型
lm = linear_model.LinearRegression()
features = ['petal_length']
X = iris[features]
y = iris['petal_width']
model = lm.fit(X,y)
#打印截距和系数
print(model.intercept_,model.coef_)
#预测petal_length为4，petal_width的值
predict = model.predict(4)
print("petal_width's value : ",predict)
#预测性能评估，5次交叉检验
scores = -cvs(lm,X,y,cv=5,scoring='neg_mean_absolute_error')
#平均绝对值误差均值
ave_score = numpy.mean(scores)
print(ave_score)


#更改为2个特征
features = ['petal_length','sepal_length']
X = iris[features]
y = iris['petal_width']
model = lm.fit(X,y)
print(model.intercept_,model.coef_)
predict = model.predict([[1,2]])
print("petal_width's value : ",predict)
scores = -cvs(lm,X,y,cv=5,scoring='neg_mean_absolute_error')
ave_score = numpy.mean(scores)
print(ave_score)


#更改为3个特征
features = ['petal_length','sepal_length','sepal_width']
X = iris[features]
y = iris['petal_width']
model = lm.fit(X,y)
print(model.intercept_,model.coef_)
predict = model.predict([[1,2,3]])
print("petal_width's value : ",predict)
scores = -cvs(lm,X,y,cv=5,scoring='neg_mean_absolute_error')
ave_score = numpy.mean(scores)
print(ave_score)
