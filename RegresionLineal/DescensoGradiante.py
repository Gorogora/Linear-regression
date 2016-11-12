import sys
from Functions import *
from Algoritmos import *


def main(argv=None):
    if argv is None or argv < 3:
        file_name = "p1_1.csv"
        folder = "ex3Data"
        num_iterations = 100
        alpha = 1.0
    else:
        file_name = argv[0]
        num_iterations = argv[1]
        alpha = argv[2]

    print("Se ejecutara el algoritmo con los siguientes datos:")
    print ("Nombre del fichero %s" %file_name)
    print("Numero de iteraciones: %d" %num_iterations)
    print("Valor de alpha: %d" %alpha)

    # load  the data
    dataX, dataY, m, num_atrib = load_data(file_name)

    #Save a copy of the unscaled features for normal equation
    data_unscaled = dataX
    data_unscaled = [[1.0] + elem for elem in dataX]

    # normalize the data
    dataX_norm, mu, sigma = normalize(dataX, num_atrib)

    #add a 1's column
    dataX = [[1.0] + elem for elem in dataX_norm]

    #inicialize theta
    theta = [0] * num_atrib

    # Gradient Descend
    # selected_learning_rate(dataX,dataY,theta,m,num_atrib,alpha,num_iterations)
    #part_one(dataX,dataY,theta,m,num_atrib,alpha,num_iterations,mu,sigma)
    #part_two(data_unscaled,dataY,num_atrib)
    #stochastic_gradient_descent(dataX, dataY, theta, m, num_atrib, alpha, num_iterations, mu, sigma)



if __name__ == "__main__":
    main()




