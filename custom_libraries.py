import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import *
from dataloader import df

class LogisticRegressionMulticlass:
    def __init__(self, learning_rate=0.01, num_iterations=1000, verbose=False):
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.verbose = verbose
        self.models = {}
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def cost_function(self, X, y, theta):
        m = len(y)
        h = self.sigmoid(np.dot(X, theta))
        cost = (-1 / m) * np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))
        return cost
    
    def gradient_descent(self, X, y):
        m, n = X.shape
        theta = np.zeros(n)
        for i in range(self.num_iterations):
            h = self.sigmoid(np.dot(X, theta))
            gradient = np.dot(X.T, (h - y)) / m
            theta -= self.learning_rate * gradient
            
            if self.verbose and i % 100 == 0:
                cost = self.cost_function(X, y, theta)
                print(f"Iteration {i}: Cost = {cost}")
        return theta
    
    def fit(self, X, y):
        unique_classes = np.unique(y)
        for c in unique_classes:
            binary_y = np.where(y == c, 1, 0)
            theta = self.gradient_descent(X, binary_y)
            self.models[c] = theta
    
    def predict_proba(self, X):
        probas = []
        for _, theta in self.models.items():
            probas.append(self.sigmoid(np.dot(X, theta)))
        return np.column_stack(probas)
    
    def predict(self, X):
        probas = self.predict_proba(X)
        return np.argmax(probas, axis=1)
    
# Train and test functions
def train_and_test_by_Logistic_R(X_train, y_train, X_test, y_test, learning_rate=0.1, num_iterations=1000):
    model = LogisticRegressionMulticlass(learning_rate=learning_rate, num_iterations=num_iterations)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = np.mean(y_pred == y_test)
    
    print("Accuracy :", accuracy)
    
    # Precision
    precision = precision_score(y_test, y_pred, average='macro')
    print("Precision:", precision)

    # Recall
    recall = recall_score(y_test, y_pred, average='macro')
    print("Recall:", recall)

    # F1 Score
    f1 = f1_score(y_test, y_pred, average='macro')
    print("F1 Score:", f1)

    # Confusion Matrix
    conf_matrix = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", conf_matrix)


