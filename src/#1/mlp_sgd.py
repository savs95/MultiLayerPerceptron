'''
Created on Mar 10, 2017

@author: savs95
'''
import math
import numpy as np

no_of_layers = input("Enter Number of Layers (excluding input layer)")
no_of_nurons = raw_input("Enter the number of neurons in each layer")
no_of_nurons = no_of_nurons.split(" ")
while(len(no_of_nurons) != no_of_layers):
    print "Please enter the correct input"
    no_of_nurons = raw_input("Enter the number of neurons in each layer")
    no_of_nurons = no_of_nurons.split(" ")
no_of_nurons = [int(i) for i in no_of_nurons]

def quadratic_cost(vec_y, vec_a, no_of_inputs):
    if len(vec_y) != len(vec_a):
        raise ValueError("Need same no of args")
    elif no_of_inputs == 0:
        raise ValueError("Fix the number_of_inputs variable to non 0")
    else:
        temp_sum = 0
        for i in range(len(vec_y)):
            temp_sum += math.pow(vec_y[i] - vec_a[i], 2)
        temp_sum = temp_sum / no_of_inputs
        return temp_sum
    
#Input to sigmoid is a scalar
def sigmoid_activation(z):
    return math.exp(-np.logaddexp(0, -z))

sigmoid_vector = np.vectorize(sigmoid_activation)

def sigmoid_derivative(z):
    return (sigmoid_activation(z)*(1 - sigmoid_activation(z)))

sigmoid_derivative_vector = np.vectorize(sigmoid_derivative)

def mlp_init(no_of_layers, no_of_nurons, no_input_neurons, expected_output, input_vector):
    weight_matrix = [] #List of weight matrices
    matrix_1 = np.random.rand(no_of_nurons[0],no_input_neurons)
    matrix_1 = np.asmatrix(matrix_1)
    weight_matrix.append(matrix_1)
    for i in range(no_of_layers-2):
        matrix_i = np.random.rand(no_of_nurons[i+1],no_of_nurons[i])
        matrix_i = np.asmatrix(matrix_i)
        weight_matrix.append(matrix_i)
    bias_vector = [] #List of bias vectors
    input_bias = np.zeros((no_input_neurons,1))
    input_bias = np.asmatrix(input_bias)
    bias_vector.append(input_bias)
    for i in range(no_of_layers-1):
        bias_i = np.random.rand(no_of_nurons[i],1)
        bias_i = np.asmatrix(bias_i)
        bias_vector.append(bias_i)
    '''
    Find all the outputs for the given weights and biases
    '''
    activation_matrix = update_activation(weight_matrix, bias_vector, input_vector)
    '''
    Calculate the error for the last layer.
    '''
    
    delta_L = np.multiply((activation_matrix[-1] - expected_output) , sigmoid_derivative_vector(calculate_zl(weight_matrix, bias_vector, input_vector, no_of_layers - 1)))
    calculate_delta(weight_matrix, bias_vector, input_vector, no_of_layers, delta_L)
    '''
    Following is the computation for feed forward equations. Function is for the full Neural Network.
    '''
def update_activation(weight_matrix, bias_vector, input_vector):
    # "yo wright"
    # weight_matrix 
    # "yo bias"
    # bias_vector
    # "yo input"
    #  input_vector
    sigma_matrix = []
    sigma_input = sigmoid_vector(input_vector)
    sigma_input = np.asmatrix(sigma_input).transpose()
    sigma_i = sigma_input
    sigma_matrix.append(sigma_i)
    for i in range(len(weight_matrix)):
        sigma_i = sigmoid_vector(weight_matrix[i]*sigma_i + bias_vector[i+1])
        sigma_matrix.append(sigma_i)
    # sigma_matrix[-1]    
    ## (sigma_matrix[-1] - np.matrix([[1],[0]]))    
    
    return sigma_matrix

def calculate_zl(weight_matrix, bias_vector, input_vector, layer_l):
    if layer_l == 0:
        return np.asmatrix(input_vector).transpose()
    else:
        sigma_input = sigmoid_vector(input_vector)
        sigma_input = np.asmatrix(sigma_input).transpose()
        sigma_i = sigma_input
        for i in range(layer_l):
            if i == layer_l -1:
                return weight_matrix[i]*sigma_i + bias_vector[i+1]
            sigma_i = sigmoid_vector(weight_matrix[i]*sigma_i + bias_vector[i+1])
            
def calculate_delta(weight_matrix, bias_vector, input_vector, no_of_layers, delta_L):
    delta_l_plus1 = delta_L
    delta_matrix = []
    delta_matrix.append(delta_l_plus1)
    for i in range(1,no_of_layers-1):
        l = no_of_layers-1-i 
        delta_l = np.multiply(weight_matrix[i].transpose() * delta_l_plus1, sigmoid_derivative_vector(calculate_zl(weight_matrix, bias_vector, input_vector, l)))
        delta_l_plus1 = delta_l 
        delta_matrix = [delta_l_plus1] +  delta_matrix
    print "from the func caldel" , delta_matrix
    return delta_matrix    

#print calculate_zl([np.matrix([[1,1],[1,1],[1,1]]),np.matrix([[2,2,2],[2,2,2]])], [np.matrix([[0],[0]]),np.matrix([[1],[1],[1]]),np.matrix([[2],[2]])], [0,0],2)
mlp_init(3, [3,2], 2, np.matrix([[1],[0]]), [0,0])
#update_activation([np.matrix([[1,1],[1,1],[1,1]]),np.matrix([[2,2,2],[2,2,2]])], [np.matrix([[0],[0]]),np.matrix([[1],[1],[1]]),np.matrix([[2],[2]])], [0,0])