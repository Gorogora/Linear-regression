from Functions import *
from matplotlib import pyplot as plt
import random
import numpy as np


def batch_gradient_descent(dataX, dataY, theta, m, num_atrib, alpha, num_iterations):
    for it in range(num_iterations):
        # update theta
        temp_theta = [0] * num_atrib
        for j in range(num_atrib):
            grad = (1 / m) * sum([(h(x_i, theta) - y_i) * x_i[j] for x_i, y_i in zip(dataX, dataY)])
            temp_theta[j] = theta[j] - alpha * grad

        theta = temp_theta

    return theta


def normal_equation(data_unscaled, dataY, num_atrib):
    x = np.matrix(data_unscaled)
    y = np.matrix(dataY)
    theta = np.zeros((num_atrib,1))

    theta = np.dot(np.dot((np.dot(x.transpose(),x)).I, x.transpose()), y.transpose())
    return theta


def stochastic_gradient_descent(dataX, dataY, theta, m, num_atrib, alpha, num_iterations, mu, sigma):
    random.seed()
    for it in range(num_iterations):
        temp_theta = [0] * num_atrib
        #update theta
        rnd = random.randint(0,num_atrib-1)
        grad = (1 / m) * sum([(h(x_i, theta) - y_i) * x_i[rnd] for x_i, y_i in zip(dataX, dataY)])
        temp_theta[rnd] = theta[rnd] - alpha * grad
        theta = temp_theta

    print("Los valores de theta son:")
    for t in range(len(theta)):
        print("Theta %d" % t + ": %d" % theta[t])

    # Calculate the predicted value
    square = (1650 - mu[0, 0]) / sigma[0, 0]
    rooms = (3 - mu[0, 1]) / sigma[0, 1]
    price = theta[0] + theta[1] * square + theta[2] * rooms
    print("El precio será de: %d" % price)  # 293187

    return theta

def selected_learning_rate(dataX, dataY, theta, m, num_atrib, alpha, num_iterations):
    alpha = [0.01, 0.03, 0.1, 0.3, 1, 1.3]
    legend = ['0.01', '0.03', '0.1', '0.3', '1', '1.3']
    plotstyle = ['b', 'r', 'g', 'k', 'b--', 'r--'];

    for i in range(len(alpha)):
        theta = [0] * num_atrib
        J = np.zeros((num_iterations, 1))
        for it in range(num_iterations):
            J[it,0] = cost_function(dataX,dataY,theta,m)

            # update theta
            temp_theta = [0] * num_atrib
            for j in range(num_atrib):
                grad = (1 / m) * sum([(h(x_i, theta) - y_i) * x_i[j] for x_i, y_i in zip(dataX, dataY)])
                temp_theta[j] = theta[j] - alpha[i] * grad

            theta = temp_theta

        plt.plot(np.arange(num_iterations), J, plotstyle[i], label=legend[i])

    plt.xlabel('Number of iterations')
    plt.ylabel('Cost function (J)')
    plt.legend()
    plt.show()


def part_one(dataX, dataY, theta, m, num_atrib, alpha, num_iterations, mu, sigma):
    theta = batch_gradient_descent(dataX, dataY, theta, m, num_atrib, alpha, num_iterations)

    print("Los valores de theta son:")
    for t in range(len(theta)):
        print("Theta %d" % t + ": %d" % theta[t])

    # Calculate the predicted value
    square = (1650 - mu[0, 0]) / sigma[0, 0]
    rooms = (3 - mu[0, 1]) / sigma[0, 1]
    price = theta[0] + theta[1] * square + theta[2] * rooms
    print("El precio será de: %d" % price)  # 293187


def part_two(data_unscaled, dataY, num_atrib):
    theta = normal_equation(data_unscaled, dataY, num_atrib)

    print("Los valores de theta son:")
    for t in range(len(theta)):
        print("Theta %d" % t + ": %d" % theta[t])

    # Calculate the predicted value
    square = 1650
    rooms = 3
    price = theta[0] + theta[1] * square + theta[2] * rooms
    print("El precio será de: %d" % price)  # 293187