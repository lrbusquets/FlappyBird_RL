from typing import Any
import numpy as np
import copy

def sqrt_nested_array(arr):
    #if isinstance(arr, list):
    #return [np.sqrt(sub_arr) for sub_arr in arr]
    n = len(arr)
    output = np.array([None] * n)
    for i, subarr in enumerate(arr):
        output[i] = np.sqrt(subarr)
    
    return output


def difference(x, y):
    return x - y

def rms(array):
    return np.sqrt(mse(array))

def mse(array):
    return np.mean(array**2)

class ReLU:
    def __call__(self, x):
        return np.maximum(0, x)
    
    def derivative(self, x):
        return 1.0 * (x > 0)

class LReLU:
    def __call__(self, x):
        return np.maximum(5e-2, x)
    
    def derivative(self, x):
        return 1.0 * (x > 0)
    
class Sigmoid:
    def __call__(self, x):
        return 1 / (1 + np.exp(-x))
    
    def derivative(self, x):
        sigmoid = self(x)
        return sigmoid * (1 - sigmoid)
    
class Softplus:
    def __call__(self, x):
        return np.log(1+np.exp(x))
    
    def derivative(self, x):
        return 1/(1+np.exp(-x))

class NeuralNetwork:
    def __init__(self, input_length, output_length, n_hidden_layers, n_neurons_array, learning_rate = 0.1, activation=ReLU(), grad_desc_method="Adam"):
        assert n_hidden_layers == len(n_neurons_array)
        
        self.activation = activation
        self.grad_desc_method = grad_desc_method

        n_neurons_array.insert(0, input_length)
        n_neurons_array.append(output_length)

        self.input_length = input_length
        self.output_length = output_length
        self.n_layers = n_hidden_layers + 1  # including output as a layer
        self.W_matrices = self.__initialise_empty_array() 
        self.b_vectors = self.__initialise_empty_array() 
        self.a_vectors = self.__initialise_empty_array(extra_elements=1)  # this one includes also input layer for fwd/backprop
        self.z_vectors = self.__initialise_empty_array()
        self.der_act_z_vectors = self.__initialise_empty_array()
        self.grad_W_matrices = self.__initialise_empty_array()
        self.grad_b_vectors = self.__initialise_empty_array()
        self.learning_rate = learning_rate

        for i in range(self.n_layers):
            self.W_matrices[i] = self.__initialise_W_array(n_neurons_array[i+1], n_neurons_array[i]) # np.random.randn(n_neurons_array[i+1], n_neurons_array[i])
            self.b_vectors[i] = np.zeros(n_neurons_array[i+1])
            self.grad_W_matrices[i] = np.zeros((n_neurons_array[i+1], n_neurons_array[i]))
            self.grad_b_vectors[i] = np.ones(n_neurons_array[i+1]) * 0.1

    def __initialise_empty_array(self, extra_elements=0):
        return np.array([None] * (self.n_layers + extra_elements))
    
    def __initialise_W_array(self, m, n, method="Xavier"):
        if self.activation==ReLU:
            method = "Kaiming"

        if method == "Xavier":
            return np.random.randn(m, n) * np.sqrt(2/(m+n))
        elif method == "Basic":
            return np.random.randn(m, n)
        elif method == "Zeros":
            return np.zeros((m,n))
        elif method == "Ones":
            return np.ones((m,n))
        elif method == "Kaiming":
            return np.random.randn(m, n) * np.sqrt(2/n)

    def evaluate(self, input):
        assert len(input) == self.input_length, "The length of the introduced input array does not coincide with the inut length declared when creating the Network object"
        self.a_vectors[0] = input

        for i in range(self.n_layers):
            self.z_vectors[i] = self.W_matrices[i].dot(input) + self.b_vectors[i]
            self.a_vectors[i+1] = self.activation(self.z_vectors[i])
            self.der_act_z_vectors[i] = self.activation.derivative(self.z_vectors[i])
            input = self.a_vectors[i+1]

        return input
    
    def backprop(self, dC_daL_vec):

        assert len(dC_daL_vec) == self.output_length, "The length of the introduced error gradient vector does not coincide with the output length declared when creating the Network object"
        dC_dal_vec = dC_daL_vec
        
        for l in reversed(range(self.n_layers)):
            dC_dzl_vec = self.der_act_z_vectors[l] * dC_dal_vec
            self.grad_b_vectors[l] += dC_dzl_vec
            self.grad_W_matrices[l] += np.outer(dC_dzl_vec, self.a_vectors[l])
            dC_dal_vec = self.W_matrices[l].T @ dC_dzl_vec

    def train(self, training_examples, target_outputs, error_factor = 2, error_function = difference, training_it = None):
        self.training_it = training_it

        n_training_examples = len(training_examples)

        errors = np.array([None] * n_training_examples)
        for i in range(n_training_examples):
            NN_output = self.evaluate(training_examples[i])
            errors[i] = error_function(NN_output, target_outputs[i])
            self.backprop(error_factor * errors[i])

        self.apply_and_reset_gradients(n_training_examples)
        
        #MSE_val = mse(errors)
        #print(f"MSE: {MSE_val}")
        return mse(errors), np.std(errors)

    def apply_and_reset_gradients(self, n_training_examples):
        self.grad_b_vectors = self.grad_b_vectors / n_training_examples
        self.grad_W_matrices = self.grad_W_matrices / n_training_examples

        self.update_params()

        for i in range(self.n_layers):
            self.grad_W_matrices[i] = np.zeros_like(self.grad_W_matrices[i])
            self.grad_b_vectors[i] = np.zeros_like(self.grad_b_vectors[i])

    def update_params(self):
        method = self.grad_desc_method
        if method=="basic":
            self.W_matrices -= self.learning_rate * self.grad_W_matrices
            self.b_vectors -= self.learning_rate * self.grad_b_vectors

        elif method=="Adam":
            assert self.training_it!=None

            beta1 = 0.9
            beta2 = 0.999
            epsilon = 1e-8

            if not hasattr(self, 'm_W_matrices_prev'):
                self.m_W_matrices_prev = np.zeros_like(self.W_matrices)
                self.v_W_matrices_prev = np.zeros_like(self.W_matrices)
                self.m_b_vectors_prev = np.zeros_like(self.b_vectors)
                self.v_b_vectors_prev = np.zeros_like(self.b_vectors)

            # Update parameters using Adam
            self.m_W_matrices = beta1 * self.m_W_matrices_prev + (1 - beta1) * self.grad_W_matrices
            self.v_W_matrices = beta2 * self.v_W_matrices_prev + (1 - beta2) * (self.grad_W_matrices ** 2)
            self.m_hat_W_matrices = self.m_W_matrices / (1 - beta1)
            self.v_hat_W_matrices = self.v_W_matrices / (1 - beta2)

            self.m_W_matrices_prev = copy.deepcopy(self.m_W_matrices)
            self.v_W_matrices_prev = copy.deepcopy(self.v_W_matrices)

            sqrt_vhat = sqrt_nested_array(self.v_hat_W_matrices) + epsilon # (np.ones_like(self.v_hat_W_matrices) *  epsilon)
            self.W_matrices -= self.learning_rate * self.m_hat_W_matrices / sqrt_vhat

            self.m_b_vectors = beta1 * self.m_b_vectors_prev + (1 - beta1) * self.grad_b_vectors
            self.v_b_vectors = beta2 * self.v_b_vectors_prev + (1 - beta2) * (self.grad_b_vectors ** 2)
            self.m_hat_b_vectors = self.m_b_vectors / (1 - beta1)
            self.v_hat_b_vectors = self.v_b_vectors / (1 - beta2)

            self.m_b_vectors_prev = copy.deepcopy(self.m_b_vectors)
            self.v_b_vectors_prev = copy.deepcopy(self.v_b_vectors)

            sqrt_vhat = sqrt_nested_array(self.v_hat_b_vectors) + epsilon #(np.ones_like(self.v_hat_b_vectors) *  epsilon)
            self.b_vectors -= self.learning_rate * self.m_hat_b_vectors / sqrt_vhat

        
    
    #def backprop(self, dC_daL_vec_array):
    #    n_training_examples = len(dC_daL_vec_array)
#
    #    for dC_daL_vec in dC_daL_vec_array:
    #        assert len(dC_daL_vec) == self.output_length, "The length of the introduced error gradient vectors does not coincide with the output length declared when creating the Network object"
    #        dC_dal_vec = dC_daL_vec
    #        
    #        for l in reversed(range(self.n_layers)):
    #            dC_dzl_vec = self.der_act_z_vectors[l] * dC_dal_vec
    #            self.grad_b_vectors[l] += dC_dzl_vec
    #            self.grad_W_matrices[l] += np.outer(dC_dzl_vec, self.a_vectors[l])
    #            dC_dal_vec = self.W_matrices[l].T @ dC_dzl_vec
#
    #    self.grad_b_vectors = self.grad_b_vectors / n_training_examples
    #    self.grad_W_matrices = self.grad_W_matrices / n_training_examples
#
    #    self.W_matrices -= self.learning_rate * self.grad_W_matrices
    #    self.b_vectors -= self.learning_rate * self.grad_b_vectors
#
    #    for i in range(self.n_layers):
    #        self.grad_W_matrices[i] = np.zeros_like(self.grad_W_matrices[i])
    #        self.grad_b_vectors[i] = np.zeros_like(self.grad_b_vectors[i])

    def get_di_dj(self, input, i, j):
        
        input_sup = input.copy()
        input_inf = input.copy()
        input_sup[j] = input_sup[j] + 1e-5
        input_inf[j] = input_inf[j] - 1e-5

        return (self.evaluate(input_sup)[i] - self.evaluate(input_inf)[i]) / (input_sup[j] - input_inf[j])


#def ReLU(z):
#    return np.maximum(0, z)
#
#def softmax(z):
#    return np.exp(z) / np.sum(np.exp(z))  # inputs > 1000 go to inf

#input_length = 3
#output_length = 1
#n_hidden_layers = 3
#n_neurons_array = [4, 3, 2] * n_hidden_layers
##
#NN = NeuralNetwork(input_length, output_length, n_hidden_layers, n_neurons_array)
##
#print(NN.evaluate([1, 2, 3]))