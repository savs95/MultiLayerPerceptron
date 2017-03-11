'''
Created on Mar 12, 2017

@author: savs95
'''

import math
import numpy as np
import random
from operator import sub

def network_input(no_of_layers,no_of_neurons):
    no_of_layers = input("Enter Number of Layers (excluding input layer)")
    no_of_neurons = raw_input("Enter the number of neurons in each layer")
    no_of_neurons = no_of_neurons.split(" ")
    while(len(no_of_neurons) != no_of_layers):
        print "Please enter the correct input"
        no_of_neurons = raw_input("Enter the number of neurons in each layer")
        no_of_neurons = no_of_neurons.split(" ")
        no_of_neurons = [int(i) for i in no_of_neurons]
    return no_of_layers, no_of_neurons    

#Input to sigmoid is a scalar
def sigmoid_activation(z):
    return math.exp(-np.logaddexp(0, -z))

sigmoid_vector = np.vectorize(sigmoid_activation)

def sigmoid_derivative(z):
    return (sigmoid_activation(z)*(1 - sigmoid_activation(z)))

sigmoid_derivative_vector = np.vectorize(sigmoid_derivative)

def mpl_init(no_of_layers, no_of_neurons, no_input_neurons, output_matrix, input_matrix, learning_rate, no_epoch):
    weight_matrix = [np.matrix([[1,1],[1,1]]), np.matrix([[2,2],[2,2]]),np.matrix([[1,1]])] #List of weight matrices
    
#     matrix_1 = np.random.normal(0,0.1,no_of_neurons[0] * no_input_neurons).reshape((no_of_neurons[0],no_input_neurons))
#     matrix_1 = np.asmatrix(matrix_1)
#     weight_matrix.append(matrix_1)
#     for i in range(no_of_layers-2):
#         matrix_i = np.random.normal(0,0.1,no_of_neurons[i+1] * no_of_neurons[i]).reshape((no_of_neurons[i+1],no_of_neurons[i]))
#         matrix_i = np.asmatrix(matrix_i)
#         weight_matrix.append(matrix_i)
    bias_vector = [np.matrix([[0],[0]]),np.matrix([[1],[1]]),np.matrix([[2],[2]]),np.matrix([[3]])] #List of bias vectors
#     input_bias = np.zeros((no_input_neurons,1))
#     input_bias = np.asmatrix(input_bias)
#     bias_vector.append(input_bias)
#     j = 0
#     for j in range(len(no_of_neurons)):
#         bias_i = np.random.normal(0,0.1,no_of_neurons[j]).reshape((no_of_neurons[j],1))
#         bias_i = np.asmatrix(bias_i)
#         bias_vector.append(bias_i)
    #We have list of matrix weights and bias vectors    
    delta_tensor = []
    activation_tensor = []
    #For each point calculate the deltas, and then list comprehension of that
    for i in range(len(input_matrix)):
        (delta_matrix, activation_matrix) = calculate_delta_x(no_of_layers, weight_matrix, bias_vector, no_input_neurons, output_matrix[i], input_matrix[i])
        delta_tensor.append(delta_matrix)
        activation_tensor.append(activation_matrix)
    print "Activation Tensor", activation_tensor
    print "Delta Tensor", delta_tensor
    #Gradient Descend Step
    for i in range(no_epoch):
        print "epoch: ", i
        sum_matrix_weight = []
        sum_matrix_bias = []
        for j in range(1,no_of_layers):
            sum_input_weight = 0
            sum_input_bias = 0
            for k in range(len(input_matrix)):
                sum_input_weight += delta_tensor[k][no_of_layers - j - 1] * activation_tensor[k][no_of_layers - j - 1].transpose()
                sum_input_bias += delta_tensor[k][no_of_layers -j - 1]
            #Loop Ends, so we have summed over all x = training examples    
            sum_input_weight = (sum_input_weight * learning_rate)/len(input_matrix)
            sum_input_bias = (sum_input_bias * learning_rate)/ len(input_matrix)
            sum_matrix_weight = [sum_input_weight]+sum_matrix_weight
            sum_matrix_bias = [sum_input_bias] + sum_matrix_bias
        sum_matrix_bias = [np.asmatrix(np.zeros((no_input_neurons,1)))] + sum_matrix_bias
        weight_matrix = map(sub,weight_matrix, sum_matrix_weight)
        print "Updated weight matrix ", weight_matrix , "sum matrix weight ", sum_matrix_weight
        bias_vector = map(sub, bias_vector, sum_matrix_bias)
        print "Updated bias vector ", bias_vector, "sum matrix bias ", sum_matrix_bias
    return(weight_matrix, bias_vector)

def calculate_delta_x(no_of_layers, weight_matrix, bias_vector, no_input_neurons, expected_output, input_vector):
    activation_matrix = update_activation(weight_matrix, bias_vector, input_vector)
    delta_L = np.multiply((activation_matrix[-1] - expected_output) , sigmoid_derivative_vector(calculate_zl(weight_matrix, bias_vector, input_vector, no_of_layers - 1)))
    delta_matrix = calculate_delta(weight_matrix, bias_vector, input_vector, no_of_layers, delta_L)
    return (delta_matrix,activation_matrix)

def update_activation(weight_matrix, bias_vector, input_vector):
    sigma_matrix = []
    sigma_input = sigmoid_vector(input_vector)
    sigma_input = np.asmatrix(sigma_input)
    sigma_i = sigma_input
    sigma_matrix.append(sigma_i)
    for i in range(len(weight_matrix)):
        sigma_i = sigmoid_vector(weight_matrix[i]*sigma_i + bias_vector[i+1])
        sigma_matrix.append(sigma_i)
    return sigma_matrix

def calculate_zl(weight_matrix, bias_vector, input_vector, layer_l):
    if layer_l == 0:
        return np.asmatrix(input_vector).transpose()
    else:
        sigma_input = sigmoid_vector(input_vector)
        sigma_input = np.asmatrix(sigma_input)
        sigma_i = sigma_input
        for i in range(layer_l):
            if i == layer_l -1:
                return weight_matrix[i]*sigma_i + bias_vector[i+1]
            else:
                sigma_i = sigmoid_vector(weight_matrix[i]*sigma_i + bias_vector[i+1])
            
def calculate_delta(weight_matrix, bias_vector, input_vector, no_of_layers, delta_L):
    delta_l_plus1 = delta_L
    delta_matrix = []
    delta_matrix.append(delta_l_plus1)
    for i in range(1,no_of_layers-1):
        l = no_of_layers-1-i
        delta_l = np.multiply(weight_matrix[l].transpose() * delta_l_plus1, sigmoid_derivative_vector(calculate_zl(weight_matrix, bias_vector, input_vector, l)))
        delta_l_plus1 = delta_l 
        delta_matrix = [delta_l_plus1] +  delta_matrix
    return delta_matrix

def generate_xor_data(no_of_points):
    input_matrix = []
    output_matrix = []
    for _ in range(no_of_points):
        var = np.matrix([[random.randrange(0, 2)],[random.randrange(0,2)]])
        op = var[1] ^ var[0]
        input_matrix.append(var)
        output_matrix.append(op)
    return(input_matrix, output_matrix)

data = ([np.matrix([[0],[0]]), np.matrix([[1],[0]])] , [np.matrix([[0]]), np.matrix([[1]])])
print data
(weight_matrix, bias_matrix) = mpl_init(4,[2,2,1],2,data[1], data[0], 1, 2)
sigma = update_activation(weight_matrix, bias_matrix, np.asmatrix([[0],[0]]))
print sigma
sigma = update_activation(weight_matrix, bias_matrix, np.asmatrix([[1],[0]]))
print sigma



#print calculate_zl([np.matrix([[1,1],[1,1],[1,1]]),np.matrix([[2,2,2],[2,2,2]])], [np.matrix([[0],[0]]),np.matrix([[1],[1],[1]]),np.matrix([[2],[2]])], [0,0],2)
#(training_data, validation_data, test_data) = load_data()
#print np.asmatrix(training_data[0][1]).transpose()
#mpl_init(7, [15,15,15,15,15,10], 786, , input_matrix, learning_rate, no_epoch)
#mlp_init(3, [3,2], 2, np.matrix([[1],[0]]), [0,0])
#update_activation([np.matrix([[1,1],[1,1],[1,1]]),np.matrix([[2,2,2],[2,2,2]])], [np.matrix([[0],[0]]),np.matrix([[1],[1],[1]]),np.matrix([[2],[2]])], [0,0])