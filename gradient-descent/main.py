# Linear Regression with Gradient descent
# y = mx + b
# m - slope
# b - y-intercept

import numpy as np
import pandas as pd

def error(points, b, m):
    err = 0
    for i in range(0,len(points)):
        x = points[i][0]
        y = points[i][1]
        err += ((y - (m * x + b)) ** 2)
    return err / len(points)

def gradient_step(points, b, m, learning_rate):
    b_gradient = 0
    m_gradient = 0
    N = len(points)
    for i in range(0,len(points)):
        x = points[i][0]
        y = points[i][1]
        b_gradient += (-2/N) * (y - (m * x + b))
        m_gradient += (-2/N) * x * (y - (m * x + b))
    b_new = b - learning_rate * b_gradient
    m_new = m - learning_rate * m_gradient
    return b_new, m_new

def gradient_descent(points, b_start, m_start, learning_rate, nr_iter):
    b = b_start
    m = m_start
    for i in range(0,nr_iter):
        b, m = gradient_step(points, b, m, learning_rate)
    return b, m

def run():
    dataset = pd.read_csv('data.csv')
    points = dataset.iloc[:, :].values
    b_start = 0
    m_start = 0
    learning_rate = 0.0001
    nr_iter = 1000
    print("Starting with b={}, m={}, error={}".format(b_start, m_start, error(points, b_start, m_start)))
    print("Running...")
    b, m = gradient_descent(points, b_start, m_start, learning_rate, nr_iter)
    print("After {} iterations b={}, m={}, error={}".format(nr_iter, b, m, error(points, b, m)))

if __name__ == '__main__':
    run()