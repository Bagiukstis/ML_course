from scipy.io import loadmat
import numpy as np
import logging
import datetime
from sklearn.base import BaseEstimator
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder
from scipy.stats import truncnorm

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

def to_sep(data, to_shuffle=True):
    x_train = []
    x_test = []
    for i in range(10):
        x_train.append(data['train%d' % i])
        x_test.append(data['test%d' % i])

    accuracy_target_classes = [i*np.ones(len(x_test[i])) for i in range(10)]

    train_concat = np.concatenate(x_train)
    test_concat = np.concatenate(x_test)

    # Train labels
    train_labels = [np.full(len(x_train[i]), i) for i in range(10)]
    train_labels = np.concatenate(train_labels)

    # Test labels
    test_labels = [np.full(len(x_test[i]), i) for i in range(10)]
    test_labels = np.concatenate(test_labels)

    if to_shuffle:
        train_shuffle, train_shuffle_labels = shuffle(train_concat, train_labels)
        return train_shuffle, train_shuffle_labels, test_concat, test_labels, accuracy_target_classes
    return train_concat, test_concat, x_test, test_labels, accuracy_target_classes

class MultilayerPerceptron(BaseEstimator):
    def __init__(self, inputLayer, hiddenLayer, outputLayer, learningRate, activation_func, max_epochs, classes_number, bias=True):
        self.inputLayer = inputLayer
        self.hiddenLayer = hiddenLayer
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
        self.weight_hidden = X.rvs((self.hiddenLayer, self.inputLayer + self.bias))

        rad = 1 / np.sqrt(self.hiddenLayer + self.bias)
        X = truncated_normal(mean=0, sd=1, low=-rad, upp=rad)
        self.weight_output = X.rvs((self.outputLayer, self.hiddenLayer + self.bias))
    def train_single_vector(self, input_vector, target_vector):
        # Adding bias to the end of input vector
        if self.bias != 0:
            input_vector = np.concatenate((input_vector, [self.bias]))
        input_vector_t = np.array(input_vector, ndmin=2).T
        target_vector_t = np.array(target_vector, ndmin=2).T

        self.output_1 = self.activation((np.dot(self.weight_hidden, input_vector_t)))

        # adding bias to the end of hidden layer
        if self.bias != 0:
            self.output_1 = np.concatenate((self.output_1, [[self.bias]]))

        self.output_2 = self.activation((np.dot(self.weight_output, self.output_1)))

        error = (target_vector - self.output_2)
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

        error_output = target_vector - self.output_2
        tmp = error_output * self.deriv(self.output_2)
        self.weight_output += self.learningRate * np.dot(tmp, self.output_1.T)
        # Updating weight Output -> Hidden

        hidden_error = np.dot(self.weight_output.T, error_output)
        tmp = hidden_error * self.deriv(self.output_1)
        if self.bias != 0:
            # removing the last row to get rid of 0's
            self.weight_hidden += self.learningRate * np.dot(tmp, inputs.T)[:-1,:]
        else:
            self.weight_hidden += self.learningRate * np.dot(tmp, inputs.T)
    def predict(self, input_vector):
        predictions = []
        for i in range(len(input_vector)):
            # Forward propagating through the network
            if self.bias != 0:
                input_vector_input = np.concatenate((input_vector[i], [self.bias]))
            input_vector_t = np.array(input_vector_input, ndmin=2).T
            self.output_1 = self.activation((np.dot(self.weight_hidden, input_vector_t)))
            if self.bias != 0:
                self.output_1 = np.concatenate((self.output_1, [[self.bias]]))

            # Argmax to decide which neuron is the biggest
            self.output_2 = self.activation((np.dot(self.weight_output, self.output_1)))
            predictions.append(np.argmax(self.output_2))
        return np.asarray(predictions)


data = loadmat('MM4_material/mnist_all.mat')

train_data, train_labels, test_data, test_labels, accuracy_target_classes = to_sep(data, to_shuffle=True)

fac = 0.99/255
train_imgs = np.asfarray(train_data) * fac + 0.01
test_imgs = np.asfarray(test_data) * fac + 0.01

onehot_encoder = OneHotEncoder(sparse=False)
train_onehot = onehot_encoder.fit_transform(train_labels.reshape(-1,1))
test_onehot = onehot_encoder.fit_transform(test_labels.reshape(-1,1))

train_onehot[train_onehot==0] = 0.01
train_onehot[train_onehot==1] = 0.99

MLP = MultilayerPerceptron(inputLayer=784, hiddenLayer=100, outputLayer=10, learningRate=0.1, max_epochs=5, activation_func='Sigmoid', classes_number=10, bias=True)
MLP_fit = MLP.fit(train_imgs, train_onehot)
prediction_labels = MLP_fit.predict(test_imgs)

accuracy = [np.sum(prediction_labels[test_labels == i] == i) / len(accuracy_target_classes[i]) * 100 for i in range(10)]
logging.info('Neural network parameters: ')
logging.info(MLP_fit.get_params())
logging.info('Accuracy: ')
logging.info(accuracy)
