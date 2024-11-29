import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from custom_libraries import Logistic_R
from utils import get_clean_dataset
# Model Score
from sklearn.metrics import mean_absolute_error,accuracy_score


df = get_clean_dataset()

X = df.iloc[:,:-1]
y = df.iloc[:, -1]

# class_counts = df["Class"].value_counts()
# print(class_counts)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

LR_model = Logistic_R()

LR_model.fit(X_train_sc, y_train)
y_pred = LR_model.predict(X_test_sc)
accuracy = np.mean(y_pred == y_test)

print('---------our model---------')
print("Mean Absolute Error :", mean_absolute_error(y_test, y_pred))
print("Accuracy :", accuracy)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_sc, y_train)
y_lr = lr.predict(X_test_sc)

print('--------inbuild model-------')
print("Mean Absolute Error :", mean_absolute_error(y_test, y_lr))
print("Accuracy:", accuracy_score(y_test, y_lr))