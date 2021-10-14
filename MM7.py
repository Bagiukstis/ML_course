from scipy.io import loadmat
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import logging
import datetime
# Logger configuration parameters
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

file_handler = logging.FileHandler('Logs/MM7.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

L=10

# A function to extract train and test sets based on the range of L
def to_sep(data):
    x_train = []
    x_test = []
    for i in range(L):
        x_train.append(data['train%d' % i])
        x_test.append(data['test%d' % i])
    return x_train, x_test

# Dataset
data = loadmat('MM4_material/mnist_all.mat')

# Train and test sets individual
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

# Different SVM approaches. Be careful with setting the L (number of datasets) to include.
# It takes an insane amount of time to build a model.
start_time = datetime.datetime.now()
svc_rbf = svm.SVC(kernel='rbf').fit(train_concat, train_labels)
logging.info('Model name: {0}, Train time: {1}, Number of data sets: {2}'.format('RBF SVC', datetime.datetime.now() - start_time, L))

start_time = datetime.datetime.now()
svc_poly = svm.SVC(kernel='poly').fit(train_concat, train_labels)
logging.info('Model name: {0}, Train time: {1}, Number of data sets: {2}'.format('Poly SVC', datetime.datetime.now() - start_time, L))

start_time = datetime.datetime.now()
svc_sigmoid = svm.SVC(kernel='sigmoid').fit(train_concat, train_labels)

# Program fails to solve svc with a linear kernel beyond 3 data sets.
# start_time = datetime.datetime.now()
# svc_linear = svm.SVC(kernel='linear').fit(train_concat, train_labels)
# logging.info('Model name: {0}, Train time: {1}, Number of data sets: {2}'.format('Linear SVC', datetime.datetime.now() - start_time, L))

logging.warning('Training complete')

# Titles for plotting
titles = ['RBF SVC', 'Poly SVC', 'Sigmoid SVC']

# Fits a prediction set to trained SVM classifier with different kernel parameters
# cnf plots a confusion matrix, while logger logs for the events.
for j, clf in enumerate((svc_rbf, svc_poly, svc_sigmoid)):
    start_time = datetime.datetime.now()
    SVM_prediction = clf.predict(test_concat)
    accuracy = [np.sum(SVM_prediction[test_labels == i] == i) / len(target_classes[i]) * 100 for i in range(L)]
    print('Done: {0}'.format(titles[j]))
    logging.info('Accuracy of {0}, is \n {1} \n With mean of {2} %'.format(titles[j], accuracy, np.mean(accuracy)))
    logging.warning('Prediction complete: {0}, prediction time: {1}'.format(titles[j], datetime.datetime.now() - start_time))
    cnf = confusion_matrix(test_labels, SVM_prediction)
    cnf_m = ConfusionMatrixDisplay(cnf)
    cnf_m.plot()
    cnf_m.ax_.set(title=titles[j])
    plt.pause(1)
plt.show()




