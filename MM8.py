'''
Multilayer perceptron

'''
from sklearn.neural_network import MLPClassifier
from scipy.io import loadmat
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import logging
import datetime

# Logger configuration parameters
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

file_handler = logging.FileHandler('Logs/MM8.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

L = 10

def to_sep(data):
    x_train = []
    x_test = []
    for i in range(L):
        x_train.append(data['train%d' % i])
        x_test.append(data['test%d' % i])
    return x_train, x_test

data = loadmat('MM4_material/mnist_all.mat')

train, test = to_sep(data)

# Train and test set concat
train_concat = np.concatenate(train)
test_concat = np.concatenate(test)

# Train labels
train_labels = [np.full(len(train[i]),i) for i in range(L)]
train_labels = np.concatenate(train_labels)

# Test labels
test_labels = [np.full(len(test[i]),i) for i in range(L)]
test_labels = np.concatenate(test_labels)

# Individual target classes
target_classes = [i*np.ones(len(test[i])) for i in range(L)]

# pipeline
param_grid = {'solver': ['sgd', 'adam', 'lbfgs'],
              'alpha': [0.0001, 0.05],
              'activation': ['logistic', 'tanh', 'relu'],
              'learning_rate': ['constant', 'invscaling', 'adaptive']}

titles = ['MLP Quasi Newton', 'MLP Stochastic gradient descent', 'MLP ADAM']

# Cross-validation to get the best params

# mlp = MLPClassifier(max_iter=100, verbose=True)
#
# # GridSearchCV to find the best hyperparameters. n_jobs = -1 activates all cores
# clf = GridSearchCV(mlp, param_grid, n_jobs=-1, cv=3)
# clf.fit(train_concat, train_labels)
#
# # Best params
# logging.info('Best parameters found:{0}\n'.format(clf.best_params_))
#
# # All results
# means = clf.cv_results_['mean_test_score']
# stds = clf.cv_results_['std_test_score']
# for mean, std, params in zip(means, stds, clf.cv_results_['params']):
#     logging.info("%0.3f (+/-%0.03f) for %r" % (mean, std * 2, params))

start_time = datetime.datetime.now()
clf = MLPClassifier(activation='relu', alpha= 0.0001, learning_rate= 'constant', solver= 'adam', max_iter=100, verbose=True).fit(train_concat, train_labels)

MLP_prediction = clf.predict(test_concat)
accuracy = [np.sum(MLP_prediction[test_labels == i] == i) / len(target_classes[i]) * 100 for i in range(L)]

logging.info('Accuracy is \n {0} \n With mean of {1} %'.format(accuracy, np.mean(accuracy)))
logging.warning('Prediction time: {0}'.format(datetime.datetime.now() - start_time))

cnf = confusion_matrix(test_labels, MLP_prediction)
cnf_m = ConfusionMatrixDisplay(cnf)
cnf_m.plot()
cnf_m.ax_.set(title='MLP prediction using best params')
plt.show()


