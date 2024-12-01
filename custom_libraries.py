import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from utils import *

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
        theta = np.random.rand(n)
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
    
def Logistic_R(learning_rate=0.1, num_iterations=1000):
    model = LogisticRegressionMulticlass(learning_rate=learning_rate, num_iterations=num_iterations)
    return model



class MulticlassSVM:
    def __init__(self, C=1.0, learning_rate=0.001, n_iters=1000):
        self.C = C
        self.lr = learning_rate
        self.n_iters = n_iters
        self.models = {}  # Store weights and biases for each class

    def _hinge_loss(self, X, y, w, b):
        n = X.shape[0]
        margins = y * (np.dot(X, w) + b)
        loss = 0.5 * np.dot(w, w) + self.C * np.sum(np.maximum(0, 1 - margins)) / n
        return loss

    def _gradient_descent(self, X, y, w, b):
        n, d = X.shape
        margins = y * (np.dot(X, w) + b)
        # Gradients
        dw = w - self.C * np.dot(X.T, (margins < 1) * y) / n
        db = -self.C * np.sum((margins < 1) * y) / n
        return dw, db

    def _train_binary_svm(self, X, y):
        n, d = X.shape
        w = np.zeros(d)  # Weights
        b = 0  # Bias

        for _ in range(self.n_iters):
            dw, db = self._gradient_descent(X, y, w, b)
            w -= self.lr * dw
            b -= self.lr * db

        return w, b

    def fit(self, X, y):
        self.classes = np.unique(y)
        for cls in self.classes:
            # Convert labels to +1 for the current class and -1 for all other classes
            y_binary = np.where(y == cls, 1, -1)
            # Train binary SVM
            w, b = self._train_binary_svm(X, y_binary)
            self.models[cls] = (w, b)

    def predict(self, X):
        # Confidence scores for each class
        confidences = np.zeros((X.shape[0], len(self.classes)))

        for i, cls in enumerate(self.classes):
            w, b = self.models[cls]
            confidences[:, i] = np.dot(X, w) + b

        # Final prediction is the class with the highest confidence
        return self.classes[np.argmax(confidences, axis=1)]


def SVM_custom(learning_rate=0.1, num_iterations=1000):
    model = MulticlassSVM()
    return model

