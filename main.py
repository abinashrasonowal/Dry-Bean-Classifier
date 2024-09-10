import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


from dataloader import df

from custom_libraries import train_and_test_by_Logistic_R
# from utils import train_test_split

# Split Train/Test
X = df.iloc[:,:-1]
y = df.iloc[:, -1]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# train_and_test_by_Logistic_R(X_train_sc,y_train,X_test_sc,y_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_lr = lr.predict(X_test)

# Model Score
from sklearn.metrics import mean_absolute_error
print("Normal Score :", lr.score(X_test, y_test))
print("Mean Absolute Error :", mean_absolute_error(y_test, y_lr))
