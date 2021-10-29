'''
Deep learning of MNIST dataset.
Currently implemented activation functions: Relu and Sigmoid
Total number of layers: 3
'''
from scipy.io import loadmat
import numpy as np
import logging
import datetime
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import truncnorm
from _overused_functions import Overused

# Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

file_handler = logging.FileHandler('Logs/MM8_manual.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

#TODO: Add evaluation metrics through epochs
#TODO: Make it viable with more layers than 1. Currently if more than 1 added - accuracy gets poorer. Find out why.


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)

class MultilayerPerceptron(BaseEstimator):
    def __init__(self, inputLayer, hiddenLayer_1, outputLayer, learningRate, activation_func, max_epochs, classes_number, bias=True):
        self.inputLayer = inputLayer
        self.hiddenLayer_1 = hiddenLayer_1
        # self.hiddenLayer_2 = hiddenLayer_2
        # self.hiddenLayer_3 = hiddenLayer_3
        self.outputLayer = outputLayer
        self.learningRate = learningRate
        self.max_epochs = max_epochs
        self.activation_func = activation_func
        self.activation = self.activation_functions[self.activation_func]
        self.deriv = self.derivative_functions[self.activation_func]
        if bias==True:
            self.bias = 1
        else:
            self.bias = 0

        self.classes_number = classes_number
        self.create_weight_matrices()
    pass

    activation_functions = {'Sigmoid': (lambda x: 1/(1 + np.exp(-x))),
                            'Relu': (lambda x: x * (x>0))}

    derivative_functions = {'Sigmoid': (lambda x: x*(1-x)),
                            'Relu': (lambda x: 1*(x>0))}

    def create_weight_matrices(self):
        rad = 1 / np.sqrt(self.inputLayer + self.bias)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weight_hidden_1 = X.rvs((self.hiddenLayer_1, self.inputLayer + self.bias))

        # rad = 1 / np.sqrt(self.hiddenLayer_1 + self.bias)
        # X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        # self.weight_hidden_2 = X.rvs((self.hiddenLayer_2, self.hiddenLayer_1 + self.bias))
        #
        # rad = 1 / np.sqrt(self.hiddenLayer_2 + self.bias)
        # X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        # self.weight_hidden_3 = X.rvs((self.hiddenLayer_3, self.hiddenLayer_2 + self.bias))

        rad = 1 / np.sqrt(self.hiddenLayer_1 + self.bias)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weight_output = X.rvs((self.outputLayer, self.hiddenLayer_1 + self.bias))
    def train_single_vector(self, input_vector, target_vector):
        # Adding bias to the end of input vector
        if self.bias != 0:
            input_vector = np.concatenate((input_vector, [self.bias]))
        input_vector_t = np.array(input_vector, ndmin=2).T
        target_vector_t = np.array(target_vector, ndmin=2).T

        self.output_hidden_1 = self.activation((np.dot(self.weight_hidden_1, input_vector_t)))

        # # adding bias to the end of hidden layer
        if self.bias != 0:
            self.output_hidden_1 = np.concatenate((self.output_hidden_1, [[self.bias]]))
        #
        # self.output_hidden_2 = self.activation((np.dot(self.weight_hidden_2, self.output_hidden_1)))
        # if self.bias != 0:
        #     self.output_hidden_2 = np.concatenate((self.output_hidden_2, [[self.bias]]))
        # self.output_hidden_3 = self.activation((np.dot(self.weight_hidden_3, self.output_hidden_2)))
        # if self.bias != 0:
        #     self.output_hidden_3 = np.concatenate((self.output_hidden_3, [[self.bias]]))



        self.output_layer = self.activation((np.dot(self.weight_output, self.output_hidden_1)))

        error = (target_vector - self.output_layer)
        total_error = sum(error)/len(input_vector)

        # Backpropagation
        self.backpropagation(input_vector_t, target_vector_t)

        return total_error
    def fit(self, train_data, test_data):
        end_epoch = 1
        total_error = 0
        while(end_epoch <=self.max_epochs):
            start_time = datetime.datetime.now()
            for i in range(len(train_data)):
                err = self.train_single_vector(train_data[i], test_data[i])
                total_error = total_error + err
                #print('Row: ',i)
            total_error = (total_error/len(train_data))
            print('Epoch {0}, Error {1}'.format(end_epoch, total_error))
            print('Epoch time: ',datetime.datetime.now()-start_time)
            end_epoch += 1
        return self
    def backpropagation(self, inputs, target_vector):
        # calculating the error_j = (output - expected) * transfer_derivative(output)
        # calculating the error = (weight_k * error_j)
        # where k is hidden_layer and j is output_layer

        error_output = target_vector - self.output_layer
        tmp = error_output * self.deriv(self.output_layer)
        self.weight_output += self.learningRate * np.dot(tmp, self.output_hidden_1.T) # output_hidden_3





        # hidden_error = np.dot(self.weight_output.T, error_output)
        # tmp = hidden_error * self.deriv(self.output_hidden_3)
        # if self.bias != 0:
        #     self.weight_hidden_3 += self.learningRate * np.dot(tmp, self.output_hidden_2.T)[:-1, :]
        #
        # hidden_error_2 = np.dot(self.weight_hidden_3.T, hidden_error[:-1, :])
        # #error_output_2 = hidden_error_2 - self.output_hidden_2
        # tmp = hidden_error_2 * self.deriv(self.output_hidden_2)
        # if self.bias != 0:
        #     self.weight_hidden_2 += self.learningRate * np.dot(tmp, self.output_hidden_1.T)[:-1,:]


        # hidden_error_3 = np.dot(self.weight_hidden_2.T, hidden_error_2[:-1, :])
        # tmp = hidden_error_3 * self.deriv(self.output_hidden_1)

        hidden_error = np.dot(self.weight_output.T, error_output)
        tmp = hidden_error * self.deriv(self.output_hidden_1)
        if self.bias != 0:
            # removing the last row to get rid of 0's
            self.weight_hidden_1 += self.learningRate * np.dot(tmp, inputs.T)[:-1, :]
        else:
            self.weight_hidden_1 += self.learningRate * np.dot(tmp, inputs.T)

    def predict(self, input_vector):
        predictions = []
        for i in range(len(input_vector)):
            # Forward propagating through the network
            if self.bias != 0:
                input_vector_input = np.concatenate((input_vector[i], [self.bias]))
            input_vector_t = np.array(input_vector_input, ndmin=2).T
            self.output_hidden_1 = self.activation((np.dot(self.weight_hidden_1, input_vector_t)))
            if self.bias != 0:
                self.output_hidden_1 = np.concatenate((self.output_hidden_1, [[self.bias]]))

            # Argmax to decide which neuron is the biggest
            # self.output_layer = self.activation((np.dot(self.weight_output, self.output_hidden_1)))
            # self.output_hidden_2 = self.activation((np.dot(self.weight_hidden_2, self.output_hidden_1)))
            # if self.bias != 0:
            #     self.output_hidden_2 = np.concatenate((self.output_hidden_2, [[self.bias]]))
            #
            # self.output_hidden_3 = self.activation((np.dot(self.weight_hidden_3, self.output_hidden_2)))
            # if self.bias != 0:
            #     self.output_hidden_3 = np.concatenate((self.output_hidden_3, [[self.bias]]))

            self.output_layer = self.activation((np.dot(self.weight_output, self.output_hidden_1)))

            predictions.append(np.argmax(self.output_layer))
        return np.asarray(predictions)


data = loadmat('MM4_material/mnist_all.mat')

train_data, train_labels, test_data, test_labels, accuracy_target_classes = Overused().to_sep(data, to_shuffle=True)

fac = 0.99/255
train_imgs = np.asfarray(train_data) * fac + 0.01
test_imgs = np.asfarray(test_data) * fac + 0.01

onehot_encoder = OneHotEncoder(sparse=False)
train_onehot = onehot_encoder.fit_transform(train_labels.reshape(-1,1))
test_onehot = onehot_encoder.fit_transform(test_labels.reshape(-1,1))

train_onehot[train_onehot==0] = 0.01
train_onehot[train_onehot==1] = 0.99

MLP = MultilayerPerceptron(inputLayer=784, hiddenLayer_1=20, outputLayer=10, learningRate=0.1, max_epochs=5, activation_func='Sigmoid', classes_number=10, bias=True)
MLP_fit = MLP.fit(train_imgs, train_onehot)
prediction_labels = MLP_fit.predict(test_imgs)

accuracy = [np.sum(prediction_labels[test_labels == i] == i) / len(accuracy_target_classes[i]) * 100 for i in range(10)]
logging.info('Neural network parameters: ')
logging.info(MLP_fit.get_params())
logging.info('Accuracy: ')
logging.info(accuracy)
