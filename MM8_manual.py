from scipy.io import loadmat
import numpy as np
import logging
import datetime
from sklearn.base import BaseEstimator
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import truncnorm
from _overused_functions import Overused
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

file_handler = logging.FileHandler('Logs/MM8_manual.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def truncated_normal(mean=0, sd=1, low=0, upp=10):
    return truncnorm((low-mean)/sd, (upp-mean)/sd, loc=mean, scale=sd)

#def confusion_matrix(input)
class MultilayerPerceptron(BaseEstimator):
    def __init__(self, structure_of_network, learningRate, activation_func, max_epochs,bias=True):
        self.structure_of_network = structure_of_network
        self.learningRate = learningRate
        self.max_epochs = max_epochs
        self.activation_func = activation_func
        self.activation = self.activation_functions[self.activation_func]
        self.deriv = self.derivative_functions[self.activation_func]
        if bias==True:
            self.bias = 1
        else:
            self.bias = 0
        self.create_weight_matrices()
    pass

    activation_functions = {'Sigmoid': (lambda x: 1/(1 + np.exp(-x))),
                            'Relu': (lambda x: x * (x>0))}

    derivative_functions = {'Sigmoid': (lambda x: x*(1-x)),
                            'Relu': (lambda x: 1*(x>0))}

    def create_weight_matrices(self):
        self.weight_matrices = []

        layer_index = 1
        total_layers = len(self.structure_of_network)
        while layer_index < total_layers:
            # iterating through the entire structure
            nodes_in = self.structure_of_network[layer_index-1]
            nodes_out = self.structure_of_network[layer_index]

            # number of random guesses
            n = (nodes_in + self.bias) * nodes_out
            rad = 1 / np.sqrt(nodes_in + self.bias)
            X = truncated_normal(mean=2, sd=1, low=-rad, upp=rad)

            #  forming a weight vector
            weight_matrix = X.rvs(n).reshape((nodes_out, nodes_in + self.bias))
            self.weight_matrices.append(weight_matrix)
            layer_index +=1

    def train(self, input_vector, target_vector):
        total_layers = len(self.structure_of_network)
        input_vector = np.array(input_vector, ndmin=2).T
        layer_index = 0

        res_vectors = [input_vector]
        while layer_index < total_layers - 1:
            # Activating every neuron in the network layer-wise
            # Input layer -> hidden_layer_1 -> hidden_layer_2 .... -> output_layer

            # when init, [-1] returns 0.
            input_vector_1 = res_vectors[-1]

            # Add bias to the vector
            if self.bias != 0:
                input_vector_1 = np.concatenate((input_vector_1, [[self.bias]]))
                res_vectors[-1] = input_vector_1

            # Steepest gradient descent for neuron
            self.output_vector = self.activation(np.dot(self.weight_matrices[layer_index], input_vector_1))

            # The output of one layer is the input to the next layer:
            res_vectors.append(self.output_vector)
            layer_index +=1

        layer_index = total_layers - 1
        target_vector = np.array(target_vector, ndmin=2).T

        output_err = target_vector - self.output_vector
        while layer_index > 0:
            # Backpropagating
            # Ouput_layer -> hidden_layer_3 -> hidden_layer_2 ... -> Input_layer

            # The output of one layer is the input to the next layer
            output_vector = res_vectors[layer_index]
            in_vector = res_vectors[layer_index-1]

            if self.bias != 0 and not layer_index==(total_layers-1):
                output_vector = output_vector[:-1,:].copy()

            # Derivative for backpropagation of neuron.
            tmp = output_err * self.deriv(output_vector)
            tmp = np.dot(tmp, in_vector.T)

            # Updated weights
            self.weight_matrices[layer_index-1] += self.learningRate * tmp

            # New output_error
            output_err = np.dot(self.weight_matrices[layer_index-1].T, output_err)
            if self.bias != 0:
                output_err = output_err[:-1,:]
            layer_index -= 1

        # Returning original error for eval of epoch
        return target_vector - self.output_vector

    def fit(self, train_data, test_data, eval=None):
        end_epoch = 1
        total_error_list = []
        while(end_epoch <=self.max_epochs):
            start_time = datetime.datetime.now()
            total_error = 0
            for i in range(len(train_data)):
                err = self.train(train_data[i], test_data[i])
                total_error = total_error + err
            total_error = (total_error/len(train_data))
            total_error_list.append(total_error)

            print('Epoch {0}'.format(end_epoch))
            print('Epoch time: ',datetime.datetime.now()-start_time)
            end_epoch += 1
        if eval == True:
            if self.max_epochs == 1:
                plt.scatter(self.max_epochs, total_error_list)
            plt.plot([i for i in range(self.max_epochs)], [i.flatten() for i in total_error_list])
            plt.show()
        return self

    def predict(self, input_vector_raw):
        total_layers = len(self.structure_of_network)
        predictions = []

        for i in range(len(input_vector_raw)):
            # Forward propagating through the network
            if self.bias != 0:
                input_vector_t = np.concatenate((input_vector_raw[i], [self.bias]))
            input_vector = np.array(input_vector_t, ndmin=2).T

            layer_index = 1
            output_vector = 0
            while layer_index < total_layers:
                # Passing the vector through every layer of the network.
                output_vector = self.activation(np.dot(self.weight_matrices[layer_index-1], input_vector))
                input_vector = output_vector

                if self.bias != 0:
                    input_vector = np.concatenate((input_vector, [[self.bias]]))
                layer_index += 1

            predictions.append(np.argmax(output_vector))
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

MLP = MultilayerPerceptron(structure_of_network=[784, 100, 50, 10], learningRate=0.1, max_epochs=100, activation_func='Sigmoid', bias=True)
MLP_fit = MLP.fit(train_imgs, train_onehot, eval=True)
prediction_labels = MLP_fit.predict(test_imgs)

accuracy = [np.sum(prediction_labels[test_labels == i] == i) / len(accuracy_target_classes[i]) * 100 for i in range(10)]

logging.info('Neural network parameters: ')
logging.info(MLP_fit.get_params())
logging.info('Accuracy: ')
logging.info(accuracy)

# Confusion matrix
cnf = confusion_matrix(test_labels, prediction_labels)
cnf_m = ConfusionMatrixDisplay(cnf)
cnf_m.plot()

plt.show()