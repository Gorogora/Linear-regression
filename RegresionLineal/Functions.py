import os, csv
import numpy as np


def get_file_path(file_name, folder=""):
    if folder == "":
        currentdirpath = os.getcwd()
    else:
        currentdirpath = os.getcwd() + "/" + folder

    file_path = os.path.join(currentdirpath, file_name)
    return file_path


def load_data(file_name):
    # get the path
    path = get_file_path(file_name)
    # open the file
    file = open(path)
    # read the file
    reader = csv.reader(file, delimiter=';')
    x = [list(map(float, line)) for line in reader]
    # x = list(reader)
    # load inputs
    temp_datax = [elem[:-1] for elem in x]  # return [[2100.0, 3.0], [1600.0, 3.0], [..., ...]]
    # load outputs
    temp_datay = [elem[-1] for elem in x]  # return [400000.0, 330000.0, 369000.0, 232000.0, ... ,]
    # number of instances of the problem
    m = len(temp_datay)
    # numbers of variables + 1
    num_atribu = len(temp_datax[0]) + 1

    return temp_datax, temp_datay, m, num_atribu  # return a tuple, we could also write (tempDataX, tempDataY, ...)


def normalize(dataset, num_atrib):
    dataset_ = np.matrix(dataset)
    mu = np.zeros((1,num_atrib-1))
    sigma = np.zeros((1,num_atrib-1))

    for i in range(num_atrib-1):
        mu[0,i] = np.mean(dataset_[:,i])
        sigma[0,i] = np.std(dataset_[:,i])
        dataset_[:,i] = (dataset_[:,i] - mu[0,i]) / sigma[0,i]

    return dataset_.tolist(), mu, sigma

# Hypothesis function
def h(x, theta):
    return sum([theta_i * x_i for theta_i, x_i in zip(theta, x)])

# Cost function
def cost_function(x, y, theta, m):
    return (0.5 / m) * sum([(h(x_i, theta) - y_i) ** 2 for x_i, y_i in zip(x, y)])
