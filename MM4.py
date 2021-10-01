'''
PCA script to reduce the dimensionality of the data set to 2 dimensions and then transform class data labels to PCA subspace.
Steps:
1. Concatinate training class labels into one array (class 5, 6 and 8)
2. Calculate the mean for every dimension and compute a covariance matrix
3. Take 2 eigenvectors with largest eigenvalues (take as many eigenvectors as dimensions that you want to reduce to)
4. Calculate PCA subspace using this formula z = w.T@(x - mean).T , where w.T is a matrix of eigenvectors. Save eigenvectors and mean of concatinated set
5. Fit test class labels (5, 6  and 8) to PCA's eigenvectors and mean.
6. Plot it and see how well did we classify it.
'''
import numpy as np
from scipy.io import loadmat
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from _overused_functions import Overused
import logging

#Setting up a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

file_handler = logging.FileHandler('Logs/MM4.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)


def to_3_classes(data, train=True):
    '''
    :param data: raw data set
    :param train: Boolean for a train set
    :return: either train classes or test classes
    '''
    if train:
        class_5 = data['train5']
        class_6 = data['train6']
        class_8 = data['train8']
        return class_5, class_6, class_8
    class_5 = data['test5']
    class_6 = data['test6']
    class_8 = data['test8']
    return class_5, class_6, class_8
def params(data):
    '''
    :param data: pass a numpy array
    :return: returns covariance matrix and mean of the data
    '''
    cov = np.cov(data.T)  #784 pivot entries
    mean = np.mean(data, axis=0) #784 dimensions
    return cov, mean
def dimensionality_reduction(x, mean, cov, dimensions):
    '''
    :param x: raw data (numpy array)
    :param mean: mean of raw data (numpy array)
    :param cov: covariance of raw data (numpy array mxm)
    :param dimensions: number of dimensions to reduce
    :return: Principal component space and largest eigenvectors
    '''
    eigen_val, eigen_vec = np.linalg.eig(cov)
    w = eigen_vec[0:,:dimensions]
    z = w.T@(x - mean).T
    return z.T, w

#Loading the data and extracting individual classes
data = loadmat('MM4_material/mnist_all.mat')
c_5, c_6, c_8 = to_3_classes(data, train=True)
c_5_t, c_6_t, c_8_t = to_3_classes(data, train=False)

#Concatinating train classes into one and then calculating the covariance matrix and mean of the set
train_concat = np.concatenate([c_5, c_6, c_8])
test_concat = np.concatenate([c_5_t, c_6_t, c_8_t])
train_concat_cov, train_concat_mean = Overused().params(train_concat)

#Establishing a PCA subspace
c_2_dim, train_eigenvector = dimensionality_reduction(x= train_concat, mean= train_concat_mean, cov= train_concat_cov, dimensions=2)

#Transforming class 5 points to PCA subspace
class_5_test = train_eigenvector.T@(c_5_t - train_concat_mean).T
class_5_test = class_5_test.T
class_5_test_mean = np.mean(class_5_test, axis=0)
class_5_test_cov = np.cov(class_5_test.T)

#Transforming class 6 points to PCA subspace
class_6_test = train_eigenvector.T@(c_6_t - train_concat_mean).T
class_6_test = class_6_test.T
class_6_test_mean = np.mean(class_6_test, axis=0)
class_6_test_cov = np.cov(class_6_test.T)

#Transforming class 8 points to PCA subspace
class_8_test = train_eigenvector.T@(c_8_t - train_concat_mean).T
class_8_test = class_8_test.T
class_8_test_mean = np.mean(class_8_test, axis=0)
class_8_test_cov = np.cov(class_8_test.T)
logging.info('Cluster 5 mean: {0}, covariance: {1}'.format(class_5_test_mean, class_5_test_cov))
logging.info('Cluster 8 mean: {0}, covarience: {1}'.format(class_8_test_mean, class_8_test_cov))
logging.info('Cluster 6 mean: {0}, covarience: {1}'.format(class_6_test_mean, class_6_test_cov))

#Multivariate dists
classes = np.array([5,6,8])

dist_5 = multivariate_normal(mean=class_5_test_mean, cov=class_5_test_cov)
dist_6 = multivariate_normal(mean=class_6_test_mean, cov=class_6_test_cov)
dist_8 = multivariate_normal(mean=class_8_test_mean, cov=class_8_test_cov)

tst_transform = train_eigenvector.T@(test_concat - train_concat_mean).T

class_5_pdf = dist_5.pdf(tst_transform.T)
class_6_pdf = dist_6.pdf(tst_transform.T)
class_8_pdf = dist_8.pdf(tst_transform.T)

pred = np.argmax(np.c_[class_5_pdf, class_6_pdf, class_8_pdf], axis=1)
pred = classes[pred]

#Create targets/classes for the test set
tst_target5 = 5*np.ones(len(c_5_t))
tst_target6 = 6*np.ones(len(c_6_t))
tst_target8 = 8*np.ones(len(c_8_t))
tst_concat = np.concatenate([tst_target5, tst_target6, tst_target8])

#Accuracies
acc5 = np.sum(pred[tst_concat == 5] == 5)/len(tst_target5) * 100
acc6 = np.sum(pred[tst_concat == 6] == 6)/len(tst_target6) * 100
acc8 = np.sum(pred[tst_concat == 8] == 8)/len(tst_target8) * 100
acc = np.sum(pred == tst_concat)/len(tst_concat) * 100

logging.shutdown()
### PLOTTING ####
plt.scatter(x=c_2_dim[:,0], y=c_2_dim[:,1], color='grey')  #PCA subspace
plt.scatter(x=class_5_test[:,0], y=class_5_test[:,1], color='blue')
plt.scatter(x=class_6_test[:,0], y=class_6_test[:,1], color='green')
plt.scatter(x=class_8_test[:,0], y=class_8_test[:,1], color='orange')
plt.show()


#### PLOTTING 3D ####
# For plotting in 3D change the dimensionality of PCA subspace to 3.

# fig = plt.figure()
# ax = plt.axes(projection='3d')
#
# ax.scatter(c_2_dim[:,0], c_2_dim[:,1], c_2_dim[:,2], cmap='blue')
# ax.scatter(class_5_test[:,0], class_5_test[:,1], class_5_test[:,2], cmap='green')
# ax.scatter(class_6_test[:,0], class_6_test[:,1], class_6_test[:,2], cmap='red')
# ax.scatter(class_8_test[:,0], class_8_test[:,1], class_8_test[:,2], cmap='yellow')
# plt.show()


