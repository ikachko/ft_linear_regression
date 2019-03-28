import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import plotly
import plotly.plotly as py
from matplotlib import style


FILE_NAME = 'data.csv'

df = pd.read_csv(FILE_NAME)

def predict(x, theta):
    return np.dot(x, theta)

def calculate_cost(x, theta, y):
    prediction = predict(x, theta)
    return ((prediction - y)**2).mean()/2


def abline(x, theta, Y):
    """Plot a line from slope and intercept"""

    y_vals = predict(x, theta)
    plt.xlim(0, 20)
    plt.ylim(-10, 60)
    plt.xlabel('No. of Rooms in the house')
    plt.ylabel('Price of house')
    plt.gca().set_aspect(0.1, adjustable='datalim')
    plt.plot(x, Y, '.', x, y_vals, '-')
    plt.show()

X = np.column_stack((np.ones(len(df['km'])), df['km']))
Y = df['price']


def gradientDescentLinearRegression(x, Y, alpha=0.047, iter=5000):
    theta0 = []
    theta1 = []
    costs = []
    theta = np.zeros(2)
    for i in range(iter):
        pred = predict(x, theta)
        t0 = theta[0] - alpha * (pred - Y).mean()
        t1 = theta[1] - alpha * ((pred - Y) * x[:, 1]).mean()

        theta = np.array([t0, t1])
        J = calculate_cost(x, theta, Y)
        theta0.append(t0)
        theta1.append(t1)
        costs.append(J)
        if i % 10 == 0:
            print(f"Iteration: {i + 1},Cost = {J},theta = {theta}")
            # abline(x, theta, Y)
    print(f'theta0 = {len(theta0)}\ntheta1 = {len(theta1)}\nCosts = {len(costs)}')

gradientDescentLinearRegression(X, Y)
