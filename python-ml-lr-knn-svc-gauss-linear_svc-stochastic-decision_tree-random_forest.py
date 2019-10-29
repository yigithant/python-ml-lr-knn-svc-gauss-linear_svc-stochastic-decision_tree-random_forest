# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in 

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.

# %% [code]
import pandas as pd
import numpy as np
import random as rnd

# visualization
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# %% [code]
train_df=pd.read_csv("/kaggle/input/trainn.csv")
test_df=pd.read_csv("/kaggle/input/testt.csv")
combine=[train_df,test_df]

# %% [code]
print(train_df.columns.values) # sütunlarýn tagleri bu þekilde belirtilir.

# %% [code]
train_df.head()

# %% [code]
train_df=train_df.drop(['Name','Ticket','Cabin'],axis=1)
test_df=test_df.drop(['Name','Ticket','Cabin'],axis=1)

# %% [code]
train_df.head()

# %% [code]
train_df=train_df.dropna()
test_df=test_df.dropna()
combine=[train_df,test_df]

for i in combine:
    i['Sex']=i['Sex'].map({'male':0, 'female':1})

train_df.head()

# %% [code]
g=sns.heatmap(train_df.corr(),annot=True)
plt.show()

# %% [code]
test_df.head()

# Fare test tarafýnda olmadýgý için train datasetten çýkarýlcak. survived ile baðlantýlý olarak
# parch sex sibsp  olarak belirlenebilir
# passengerId ye gerek yok.

# %% [code]
test_df=test_df.drop(['PassengerId'],axis=1)
train_df=train_df.drop(['PassengerId', 'Fare'],axis=1)

# %% [code]
train_df[['Pclass','Survived']].groupby(['Pclass']).mean().sort_values(by='Survived')

# %% [code]
train_df[['Sex','Survived']].groupby(['Sex']).mean().sort_values(by='Survived')

# %% [code]
# yaþ aralýðý fazla olacaðý için gruplama yapýyorum.

combine=[train_df, test_df]

for i in combine:
    i.loc[(i['Age']>=0) &(i['Age']<=18), 'Age']=0
    i.loc[(i['Age']>18) &(i['Age']<=45), 'Age']=1
    i.loc[(i['Age']>45) &(i['Age']<=65), 'Age']=2
    i.loc[i['Age']>65, 'Age']=3

# 0-18, 18-45, 45-65 ve 65 üzeri olarak gruplandýrdým.

# %% [code]
train_df[['Age','Survived']].groupby(['Age']).mean().sort_values(by='Survived')

# %% [code]
for i in combine:
    i['Embarked']=i['Embarked'].map({'S':0, 'C':1, 'Q':2 })
    
train_df[['Embarked','Survived']].groupby(['Embarked']).mean().sort_values(by='Survived')

# %% [code]
train_df[['SibSp','Survived']].groupby(['SibSp']).mean().sort_values(by='Survived')

# %% [code]
print('train_df shape : ' ,train_df.shape)
print('\ntest_df shape : ', test_df.shape)

# %% [code]
#logistic regression

x_train=train_df.drop('Survived',axis=1)
y_train=train_df['Survived']
x_test=test_df

# %% [code]
x_test.head()

# %% [code]
x_test=x_test.drop('Fare',axis=1)

print('x_train shape : ' ,x_train.shape)
print('\ny_train shape : ', y_train.shape)
print('\nx_test shape : ',x_test.shape)

# %% [code]
# logictic regression

lr=LogisticRegression()
lr.fit(x_train,y_train)
y_pred_lr=lr.predict(x_test)
print("logistic regression score : ", lr.score(x_train,y_train))

# %% [code]
# svc - support vector machines

svc=SVC()
svc.fit(x_train,y_train)
y_pred_svc=svc.predict(x_test)
print("svc score : ", svc.score(x_train,y_train))

# %% [code]
# knn

knn=KNeighborsClassifier(n_neighbors=4)
knn.fit(x_train,y_train)
y_pred_knn=knn.predict(x_test)
print("knn score : ", knn.score(x_train,y_train))

# %% [code]
# gauss

gauss=GaussianNB()
gauss.fit(x_train,y_train)
y_pred_gauss=gauss.predict(x_test)
print("gauss score : ", gauss.score(x_train,y_train))

# %% [code]
# linear svc

lr_svc=LinearSVC()
lr_svc.fit(x_train,y_train)
y_pred_lr_svc=lr_svc.predict(x_test)
print("linear svc score : ",lr_svc.score(x_train,y_train))

# %% [code]
# stochastic

sgdc=SGDClassifier()
sgdc.fit(x_train,y_train)
y_pred_sgdc=sgdc.predict(x_test)
print("sgdc score : ", sgdc.score(x_train,y_train))

# %% [code]
# decision tree

dt=DecisionTreeClassifier()
dt.fit(x_train,y_train)
y_pred_dt=dt.predict(x_test)
print("dt score : ", dt.score(x_train,y_train))

# %% [code]
#random forest

rf=RandomForestClassifier(n_estimators=100)
rf.fit(x_train,y_train)
y_pred_rf=rf.predict(x_test)
print("random forest score : ", rf.score(x_train,y_train))

# %% [code]
x_test['lr']=y_pred_lr

# %% [code]
x_test[['Embarked','lr']].groupby(['Embarked']).mean().sort_values(by='lr')

# %% [code]
x_test[['Pclass','lr']].groupby(['Pclass']).mean().sort_values(by='lr')

# %% [code]
x_test[['Sex','lr']].groupby(['Sex']).mean().sort_values(by='lr')

# %% [code]
x_test[['Age','lr']].groupby(['Age']).mean().sort_values(by='lr')

# %% [code]
x_test[['SibSp','lr']].groupby(['SibSp']).mean().sort_values(by='lr')

# %% [code]
x_test['svc']=y_pred_svc

# %% [code]
x_test[['Embarked','svc']].groupby(['Embarked']).mean().sort_values(by='svc')

# %% [code]
x_test[['Pclass','svc']].groupby(['Pclass']).mean().sort_values(by='svc')

# %% [code]
x_test[['Sex','svc']].groupby(['Sex']).mean().sort_values(by='svc')

# %% [code]
x_test[['Age','svc']].groupby(['Age']).mean().sort_values(by='svc')

# %% [code]
x_test[['SibSp','svc']].groupby(['SibSp']).mean().sort_values(by='svc')