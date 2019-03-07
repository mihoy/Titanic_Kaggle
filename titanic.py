# -*- coding: utf-8 -*-

# data analysis and wrangling
import pandas as pd
import numpy as np
import random as rnd


# machine learning
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron
from sklearn.linear_model import SGDClassifier
from sklearn.tree import DecisionTreeClassifier

# machine learning tools
from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix



train_df = pd.read_csv('input/train.csv')
test_df = pd.read_csv('input/test.csv')
combine = [train_df, test_df]

# print(train_df.columns.values)
# print(test_df.columns.values)

# print(train_df.head(5))
# print(test_df.head(5))

# print(train_df.info())
# print(test_df.info())
# for df in combine:
#     print(df.describe(include=[np.object]))
#print(test_df.describe())

#print(train_df[['Cabin', 'Survived']].groupby(['Cabin'], as_index=False).mean().sort_values(by='Survived', ascending=False))

# print("Before", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

train_df = train_df.drop(['Ticket', 'Cabin'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin'], axis=1)
combine = [train_df, test_df]

# print("After", train_df.shape, test_df.shape, combine[0].shape, combine[1].shape)

for dataset in combine:
    dataset['Title'] = dataset.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

for dataset in combine:
    dataset['Title'] = dataset['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}
for dataset in combine:
    dataset['Title'] = dataset['Title'].map(title_mapping)
    dataset['Title'] = dataset['Title'].fillna(0)

print(train_df.head())   
    
print(train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

train_df = train_df.drop(['Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Name'], axis=1)
combine = [train_df, test_df]
train_df.shape, test_df.shape

for dataset in combine:
    dataset['Sex'] = dataset['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    

for dataset in combine:
    dataset["Age"].fillna(train_df.Age.median(), inplace=True) 
    dataset['Embarked'] = dataset['Embarked'].fillna('S')
    dataset['Embarked'] = dataset['Embarked'].map({'C':1, 'S':2, 'Q':3}).astype(int)

train_df.info()


random_state=5

# データ検証用
#
#x_train = トレインデータの学習データ
#x_val = テストデータの　学習データ
#Y_train = とれいんデータの答え
#Y_val = テストデータの答え

X_train, X_val, Y_train, Y_val = train_test_split(combine[0].drop(["Survived"], axis=1), combine[0]["Survived"], train_size=0.8, random_state=random_state)



# 確率的勾配降下法
sgd = SGDClassifier(random_state = random_state)
sgd.fit(X_train, Y_train)
Y_train_pred = sgd.predict(X_train)
Y_val_pred = sgd.predict(X_val)
acc_sgd = round(sgd.score(X_train, Y_train) * 100, 2)
print('Accuracy on Training Set: {:.3f}'.format(accuracy_score(Y_train, Y_train_pred)))
print('Accuracy on Validation Set: {:.3f}'.format(accuracy_score(Y_val, Y_val_pred)))
print("SGD={}".format(acc_sgd))

# ロジスティック回帰
logreg = LogisticRegression(random_state = random_state)
logreg.fit(X_train, Y_train)
Y_train_pred = logreg.predict(X_train)
Y_val_pred = logreg.predict(X_val)
acc_log = round(logreg.score(X_train, Y_train) * 100, 2)
cm = confusion_matrix(Y_val, Y_val_pred)
print(cm)
print('Accuracy on Training Set: {:.3f}'.format(accuracy_score(Y_train, Y_train_pred)))
print('Accuracy on Validation Set: {:.3f}'.format(accuracy_score(Y_val, Y_val_pred)))
print("ロジスティック回帰={}".format(acc_log))

# Support Vector Machines
svc = SVC()
svc.fit(X_train, Y_train)
Y_train_pred = svc.predict(X_train)
Y_val_pred = svc.predict(X_val)
acc_svc = round(svc.score(X_train, Y_train) * 100, 2)
cm = confusion_matrix(Y_val, Y_val_pred)
print(cm)
print('Accuracy on Training Set: {:.3f}'.format(accuracy_score(Y_train, Y_train_pred)))
print('Accuracy on Validation Set: {:.3f}'.format(accuracy_score(Y_val, Y_val_pred)))
print("SVC={}".format(acc_svc))

# KNN(k近傍法)
knn = KNeighborsClassifier(n_neighbors = 3)
knn.fit(X_train, Y_train)
Y_train_pred = knn.predict(X_train)
Y_val_pred = knn.predict(X_val)
acc_knn = round(knn.score(X_train, Y_train) * 100, 2)
cm = confusion_matrix(Y_val, Y_val_pred)
print(cm)
print('Accuracy on Training Set: {:.3f}'.format(accuracy_score(Y_train, Y_train_pred)))
print('Accuracy on Validation Set: {:.3f}'.format(accuracy_score(Y_val, Y_val_pred)))
print("knn={}".format(acc_knn))

# # Decision Tree
# decision_tree = DecisionTreeClassifier(random_state = random_state,criterion='entropy', max_depth=10, min_samples_leaf=2)
# decision_tree.fit(X_train, Y_train)
# Y_pred_decision_tree = decision_tree.predict(X_test)
# acc_decision_tree = round(decision_tree.score(X_train, Y_train) * 100, 2)
# print("決定木={}".format(acc_decision_tree))


# Random Forest
random_forest = RandomForestClassifier(n_estimators=100,random_state = random_state,criterion='entropy',max_depth=25, min_samples_leaf=1)
random_forest.fit(X_train, Y_train)
Y_train_pred = random_forest.predict(X_train)
Y_val_pred = random_forest.predict(X_val)
acc_random_forest = round(random_forest.score(X_train, Y_train) * 100, 2)
cm = confusion_matrix(Y_val, Y_val_pred)
print(cm)
print('Accuracy on Training Set: {:.3f}'.format(accuracy_score(Y_train, Y_train_pred)))
print('Accuracy on Validation Set: {:.3f}'.format(accuracy_score(Y_val, Y_val_pred)))
print("ランダムフォレスト={}".format(acc_random_forest))

# # Perceptron
# perceptron = Perceptron(random_state = random_state)
# perceptron.fit(X_train, Y_train)
# Y_pred_perceptron = perceptron.predict(X_test)
# acc_perceptron = round(perceptron.score(X_train, Y_train) * 100, 2)
# print("パーセプトロン={}".format(acc_perceptron))

# # MLP(多層パーセプトロン)
# mlp = MLPClassifier(solver="sgd",random_state = random_state,max_iter=10000)
# mlp.fit(X_train, Y_train)
# # 結果表示
# Y_train_pred = mlp.predict(X_train)
# Y_val_pred = mlp.predict(X_val)
# acc_mlp = round(mlp.score(X_train, Y_train) * 100, 2)
# cm = confusion_matrix(Y_val, Y_val_pred)
# print(cm)
# print('Accuracy on Training Set: {:.3f}'.format(accuracy_score(Y_train, Y_train_pred)))
# print('Accuracy on Validation Set: {:.3f}'.format(accuracy_score(Y_val, Y_val_pred)))
# print("多層パーセプトロン={}".format(acc_mlp))

