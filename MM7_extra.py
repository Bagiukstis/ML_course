'''
This exercise features SVM classification on PCA dimensionality deduced data sets.
PCA reduces the data sets from 784 dimensions to only 2 dimensions.
Then, SVM is performed to do classification.
The results are presented in confusion matrix form together as countour plt to visualize points and decision boundaries.

Result: Very inaccurate classification.
'''
from scipy.io import loadmat
import numpy as np
from sklearn import svm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.decomposition import PCA
import logging
import datetime


logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

file_handler = logging.FileHandler('Logs/MM7_extra.log')
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

# PCA to reduce to 2D
pca = PCA(n_components=2)
train_reduced = pca.fit_transform(train_concat, train_labels)
test_reduced = pca.fit_transform(test_concat, test_labels)

# SVM to classify

start_time = datetime.datetime.now()
svc_rbf = svm.SVC(kernel='rbf').fit(train_reduced, train_labels)
logging.info('Model name: {0}, Train time: {1}, Number of data sets: {2}'.format('RBF SVC', datetime.datetime.now() - start_time, L))

start_time = datetime.datetime.now()
svc_poly = svm.SVC(kernel='poly').fit(train_reduced, train_labels)
logging.info('Model name: {0}, Train time: {1}, Number of data sets: {2}'.format('Poly SVC', datetime.datetime.now() - start_time, L))

start_time = datetime.datetime.now()
svc_sigmoid = svm.SVC(kernel='sigmoid').fit(train_reduced, train_labels)

logging.warning('Training complete')

titles = ['RBF SVC', 'Poly SVC', 'Sigmoid SVC']

# Creating a mesh grid for plotting points (took it from a guide)
# Uncomment blow if boundary plt is desired

# x_min, x_max = test_reduced[:, 0].min() - 1, test_reduced[:, 0].max() + 1
# y_min, y_max = test_reduced[:, 1].min() - 1, test_reduced[:, 1].max() + 1
# xx, yy = np.meshgrid(np.arange(x_min, x_max, 1),
#                      np.arange(y_min, y_max, 1))

for j, clf in enumerate((svc_rbf, svc_poly, svc_sigmoid)):
    start_time = datetime.datetime.now()
    SVM_prediction = clf.predict(test_reduced)
    accuracy = [np.sum(SVM_prediction[test_labels == i] == i) / len(target_classes[i]) * 100 for i in range(L)]
    logging.info('Accuracy of {0}, is \n {1} \n With mean of {2} %'.format(titles[j], accuracy, np.mean(accuracy)))
    cnf = confusion_matrix(test_labels, SVM_prediction)
    cnf_m = ConfusionMatrixDisplay(cnf)
    cnf_m.plot()
    cnf_m.ax_.set(title=titles[j])

    # Uncomment below if boundary plt is desired

    # fig, ax = plt.subplots()
    # Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
    # # Reshaping labels to test points
    # Z = Z.reshape(xx.shape)

    #Color area:
    # ax.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.8)
    #
    # #Testing points
    # ax.scatter(test_reduced[:, 0], test_reduced[:, 1], c=test_labels, cmap=plt.cm.coolwarm)
    # ax.set_title(titles[j])


    logging.warning(
        'Prediction complete: {0}, prediction time: {1}'.format(titles[j], datetime.datetime.now() - start_time))
    print('Done: {0}'.format(titles[j]))
    plt.pause(1)
plt.show()