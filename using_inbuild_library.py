import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataloader import df
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error,accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
# from utils import *
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from utils import get_clean_dataset

df = get_clean_dataset()

X = df.iloc[:,:-1]
y = df.iloc[:, -1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_lr = lr.predict(X_test)

print("----------LR unscaled ---------")
print("Mean Absolute Error :", mean_absolute_error(y_test, y_lr))
print("Accuracy:", accuracy_score(y_test, y_lr))

# LogisticRegressor - Scaled
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_sc, y_train)
y_lr = lr.predict(X_test_sc)

print("----------LR scaled ----------")
print("Mean Absolute Error :", mean_absolute_error(y_test, y_lr))
print("Accuracy:", accuracy_score(y_test, y_lr))


# -------------------SVM----------------
from sklearn.svm import SVC
# Create SVM Classifier - not scaled
svm_classifier = SVC(decision_function_shape='ovr')
svm_classifier.fit(X_train, y_train)
y_svm = svm_classifier.predict(X_test)

# Model Score
print("----------SVM unscaled --------")
print("SVM Classifier Score:", svm_classifier.score(X_test, y_test))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_svm))

# Create SVM Classifier scaled
svm_classifier2 = SVC(decision_function_shape='ovr') 
svm_classifier2.fit(X_train_sc, y_train)
y_svm = svm_classifier2.predict(X_test_sc)

print("----------SVM scaled ---------")
print("SVM Classifier Score:", svm_classifier2.score(X_test_sc, y_test))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_svm))


# ----------------- Removing outlier using Kmean ----------------
indicators = df.columns[:-1]
km = KMeans(n_clusters = 2)
km.fit(df[indicators])
labels = km.labels_
df_km = df[labels == 1]
df_km.reset_index(inplace = True)

from sklearn.preprocessing  import LabelEncoder
le = LabelEncoder()
df_km = df_km.copy()
df_km["Class"] = le.fit_transform(df_km["Class"])
X = df_km.iloc[:,:-1]
y = df_km.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

#-------------------- Logistic Regressor -------------
lr = LogisticRegression()
lr.fit(X_train_sc, y_train)
y_lr = lr.predict(X_test_sc)

# Model Score
print("------------------LR ----------")
print("Normal Score :", lr.score(X_test_sc, y_test))
print("Mean Absolute Error :", mean_absolute_error(y_test, y_lr))
conf_matrix = confusion_matrix(y_test, y_lr)
print("Confusion Matrix:\n", conf_matrix)

# ---------------------- SVM Classifier------------
svm_classifier2 = SVC()
svm_classifier2.fit(X_train_sc, y_train)
y_svm = svm_classifier2.predict(X_test_sc)

print("---------------SVM----------------")
print("SVM Classifier Score:", svm_classifier2.score(X_test_sc, y_test))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_svm))
conf_matrix = confusion_matrix(y_test, y_svm)
print("Confusion Matrix:\n", conf_matrix)
