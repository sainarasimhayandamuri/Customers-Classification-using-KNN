#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 16 15:15:26 2021

@author: sainarasimhaprasad
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

data = pd.read_csv("/home/sainarasimhaprasad/Desktop/Retail_Demo.csv/Retail_Demo.csv")

data.isnull().sum()
data = data.dropna()
data.columns

dataset = data.drop(["date"], axis = 1)


lb = LabelEncoder()

dataset["gender"] = lb.fit_transform(dataset["gender"])
dataset["marital_status"] = lb.fit_transform(dataset["marital_status"])
dataset["rewards_program"] = lb.fit_transform(dataset["rewards_program"])
dataset["point_redemption_method"] = lb.fit_transform(dataset["point_redemption_method"])
dataset["brand"] = lb.fit_transform(dataset["brand"])
dataset["region"] = lb.fit_transform(dataset["region"])
dataset["channel"] = lb.fit_transform(dataset["channel"])
dataset["department"] = lb.fit_transform(dataset["department"])
dataset["item"] = lb.fit_transform(dataset["item"])
dataset["payment_method"] = lb.fit_transform(dataset["payment_method"])
dataset["age_grouping"] = lb.fit_transform(dataset["age_grouping"])


dataset = dataset.iloc[:,[0,2,3,4,8,9,10,12,14,15,16,17,19,20,1,7,5,13,18,21,11,6]]

dataset = dataset.drop(["customer_id"], axis = 1)
dataset = dataset.drop(["zip_code"], axis = 1)
dataset = dataset.drop(["state"], axis = 1)


cols_to_norm = ['rewards_points','cost','satisfaction_score','revenue']
dataset[cols_to_norm] = dataset[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

X = np.array(dataset.iloc[:,0:18]) # Predictors 
Y = np.array(dataset['customer_category']) # Target 

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors = 5)
knn.fit(X_train, Y_train)

pred = knn.predict(X_test)
pred

#Evaluate the model
from sklearn.metrics import accuracy_score
print(accuracy_score(Y_test, pred))
pd.crosstab(Y_test, pred, rownames = ['Actual'], colnames= ['Predictions'])

# error on train data
pred_train = knn.predict(X_train)
pred_train

print(accuracy_score(Y_train, pred_train))
pd.crosstab(Y_train, pred_train, rownames=['Actual'], colnames = ['Predictions'])

# creating empty list variable 
acc = []

# running KNN algorithm for 3 to 50 nearest neighbours(odd numbers) and 
# storing the accuracy values

for i in range(3,50,2):
    neigh = KNeighborsClassifier(n_neighbors=i)
    neigh.fit(X_train, Y_train)
    train_acc = np.mean(neigh.predict(X_train) == Y_train)
    test_acc = np.mean(neigh.predict(X_test) == Y_test)
    acc.append([train_acc, test_acc])


import matplotlib.pyplot as plt 

#train accuracy plot
plt.plot(np.arange(3,50,2),[i[0] for i in acc],"ro-")

# test accuracy plot
plt.plot(np.arange(3,50,2),[i[1] for i in acc],"bo-")
