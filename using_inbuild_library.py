import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from dataloader import df
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import *

# Split Train/Test
X = df.iloc[:,:-1]
y = df.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)


sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)
# ---------------- Logistic Regressor----------------
# LogisticRegressor - Not Scaled
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train, y_train)
y_lr = lr.predict(X_test)

# Model Score
from sklearn.metrics import mean_absolute_error
print("Normal Score :", lr.score(X_test, y_test))
print("Mean Absolute Error :", mean_absolute_error(y_test, y_lr))

# LogisticRegressor - Scaled
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_sc, y_train)
y_lr = lr.predict(X_test_sc)

# Model Score
from sklearn.metrics import mean_absolute_error
print("Normal Score :", lr.score(X_test_sc, y_test))
print("Mean Absolute Error :", mean_absolute_error(y_test, y_lr))

# ------------- Random forest----------------
# RandomForestRegressor - Not Scaled
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train, y_train)
y_rfr = rfr.predict(X_test)

# Model Score
from sklearn.metrics import mean_absolute_error
print("Normal Score :", rfr.score(X_test, y_test))
print("Mean Absolute Error :", mean_absolute_error(y_test, y_rfr))

# RandomForestRegressor - Scaled
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor()
rfr.fit(X_train_sc, y_train)
y_rfr = rfr.predict(X_test_sc)

# Model Score
from sklearn.metrics import mean_absolute_error
print("Normal Score :", rfr.score(X_test_sc, y_test))
print("Mean Absolute Error :", mean_absolute_error(y_test, y_rfr))

# ------------------------Decision Tree --------------
from sklearn.tree import DecisionTreeClassifier

# Create Decision Tree Classifier - not scaled
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train, y_train)
y_dt = dt_classifier.predict(X_test)

# Model Score
print("Decision Tree Classifier Score:", dt_classifier.score(X_test, y_test))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_dt))

# Create Decision Tree Classifier - scaled
dt_classifier = DecisionTreeClassifier()
dt_classifier.fit(X_train_sc, y_train)
y_dt = dt_classifier.predict(X_test_sc)

# Model Score
print("Decision Tree Classifier Score:", dt_classifier.score(X_test_sc, y_test))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_dt))

# -------------------SVM----------------
from sklearn.svm import SVC

# Create SVM Classifier - not scaled
svm_classifier = SVC()
svm_classifier.fit(X_train, y_train)
y_svm = svm_classifier.predict(X_test)

# Model Score
print("SVM Classifier Score:", svm_classifier.score(X_test, y_test))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_svm))

# Create SVM Classifier scaled
svm_classifier2 = SVC() 
svm_classifier2.fit(X_train_sc, y_train)
y_svm = svm_classifier2.predict(X_test_sc)

# Model Score
print("SVM Classifier Score:", svm_classifier2.score(X_test_sc, y_test))
print("Mean Absolute Error:", mean_absolute_error(y_test, y_svm))


indicators = df.columns[:-1]
km = KMeans(n_clusters = 2)
km.fit(df[indicators])
labels = km.labels_
df_km = df[labels == 1]
df_km.reset_index(inplace = True)

# Correlation of Dry Bran with Class
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
df_km["Class"] = le.fit_transform(df_km["Class"])

# Split Train/Test
X = df_km.iloc[:,:-1]
y = df_km.iloc[:, -1]
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

#-------------------- LogisticRegressor - Scaled-------------
from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
lr.fit(X_train_sc, y_train)
y_lr = lr.predict(X_test_sc)

# Model Score
from sklearn.metrics import mean_absolute_error
print("Normal Score :", lr.score(X_test_sc, y_test))
print("Mean Absolute Error :", mean_absolute_error(y_test, y_lr))

# Precision
precision = precision_score(y_test, y_lr, average='macro')
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_lr, average='macro')
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_test, y_lr, average='macro')
print("F1 Score:", f1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_lr)
print("Confusion Matrix:\n", conf_matrix)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# -------------------- Decision Tree Regressor--------------
dtr = DecisionTreeRegressor()

# Train Decision Tree Regressor
dtr.fit(X_train_sc, y_train)

# Predict using Decision Tree Regressor
y_dtr = dtr.predict(X_test_sc)

# Model Score
print("Normal Score :", dtr.score(X_test_sc, y_test))
print("Mean Absolute Error :", mean_absolute_error(y_test, y_dtr))

# Mean Squared Error
mse = mean_squared_error(y_test, y_dtr)
print("Mean Squared Error:", mse)

# R-squared
r2 = r2_score(y_test, y_dtr)
print("R-squared:", r2)


# ---------------------- SVM Classifier------------
svm_classifier2 = SVC()

# Train SVM Classifier scaled
svm_classifier2.fit(X_train_sc, y_train)

# Predict using SVM Classifier
y_svm = svm_classifier2.predict(X_test_sc)

# Model Score
print("SVM Classifier Score:", svm_classifier2.score(X_test_sc, y_test))

# Mean Absolute Error
print("Mean Absolute Error:", mean_absolute_error(y_test, y_svm))

# Precision
precision = precision_score(y_test, y_svm, average='macro')
print("Precision:", precision)

# Recall
recall = recall_score(y_test, y_svm, average='macro')
print("Recall:", recall)

# F1 Score
f1 = f1_score(y_test, y_svm, average='macro')
print("F1 Score:", f1)

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_svm)
print("Confusion Matrix:\n", conf_matrix)
