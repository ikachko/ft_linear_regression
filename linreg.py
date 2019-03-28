import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

FILE_NAME = 'data.csv'

df = pd.read_csv(FILE_NAME)

class LinearRegression:
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def plot_data(self):
        if None not in [self.X, self.y]:
            plt.scatter(self.X, self.y, color='b', marker='o', s=30)
            plt.xlabel('X')
            plt.ylabel('y')
            plt.show()

    def predict(self, x, teta):
        return teta[0] + teta[1] * x

    def fit(self, X, y, teta, epochs=500, learning_rate=0.1):

    def MSE(self, X, Y, f, teta):
        distance = 0
        for xi, yi in zip(X, Y):
            distance += (yi - f(xi, teta)) ** 2
        return distance

    def plot_regression_line(self, X, y, teta):
        plt.scatter(X, y, color="m",
                    marker="o", s=30)
        y_pred = self.predict(X, teta)
        plt.plot(X, y_pred, color='g')

        plt.xlabel('x')
        plt.ylabel('y')
        plt.show()