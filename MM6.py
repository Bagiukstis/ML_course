'''
A script for an LDA exercise. The goal is to reduce the dimensionality to either 2 or 9 dimensions by using the LDA method.
When comparing PCA and LDA methods - LDA returns a better overall accuracy. But it is also a bit more tricky to compute it.
'''
from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

C = 10

def to_sep(data):
    x_train = []
    x_test = []
    for i in range(C):
        x_train.append(data['train%d' % i])
        x_test.append(data['test%d' % i])
    return x_train, x_test
def reduce_dim(eigen_vec, data, dim=2):
    w = eigen_vec[:dim].T
    LDA = data.dot(w)
    return LDA
def normal_dist(test_data, means, covs):
    container = []
    test_data = np.concatenate([test_data[j].T for j in range(len(means))])
    for i in range(len(means)):
        container.append(multivariate_normal.pdf(test_data, means[i], covs[i]))
    container = np.c_[tuple(container)]
    labels = np.argmax(container, axis=1)
    return labels
def params(lda_data):
    means = []
    covs = []
    for i in range(len(lda_data)):
        means.append(np.mean(lda_data[i].T, axis =0))
        covs.append(np.cov(lda_data[i]))
    return means, covs
def train_test_return(train, test, eigenvec, dim=2):
    if dim==2:
        train_set = []
        test_set = []
        l2_vec = np.asarray([eigenvec[0], eigenvec[1]])
        for i in range(len(train)):
            train_set.append(np.dot(l2_vec, train[i].T))
            test_set.append(np.dot(l2_vec, test[i].T))
        return train_set, test_set
    if dim==9:
        train_set = []
        test_set = []
        for i in range(len(train)):
            train_set.append(np.dot(eigenvec, train[i].T))
            test_set.append(np.dot(eigenvec, test[i].T))
        return train_set, test_set
    else:
        return [], []
def sortkey(x):
    # Define that we are only sorting by the size of the first value (eigenvalue), thereby also sorting indexes
    return x[0]

data = loadmat('MM4_material/mnist_all.mat')

# Train and test sets
train, test = to_sep(data)

# Mean container
mean_container = [np.mean(i, axis=0) for i in train]

# Common mean
miu = sum(mean_container)/10

# Lenght of every single dataset
n = [len(x) for x in train]

# Between matrix calculation
S_b = np.zeros((784, 784))
for i in range(C):
    S_b += n[i] * ((mean_container[i]-miu) * ((mean_container[i]-miu))[:,None])

# Within matrix calculation:
S_w  = np.zeros((784, 784))
for i in range(C):
    cov = np.cov(train[i].T)
    S_w = S_w + cov

# Computing eigenvalues and eigenvectors
eig_val, eig_vec = np.linalg.eig(np.linalg.pinv(S_w).dot(S_b))

# Find index for 9 largest values
max_idx = ind = np.argpartition(eig_val, -9)[-9:]

# Placeholder table for indices and values
tab = [[eig_val[i], i] for i in max_idx]

# Do the sorting, reverse order to have descending (largest eigval first)
s_eigvals = sorted(tab,key=sortkey,reverse=True)

# Create projection matrix to 9 dimensions based on indices of largest eigenvalues
l9_eigvecs = [eig_vec[:,i[1]] for i in s_eigvals]

# Computing train and test LDA's
train_lda, test_lda = train_test_return(train, test, l9_eigvecs, dim=9)

# Computing true test labels
test_labels = [np.full(len(test[i]),i) for i in range(C)]
test_labels = np.concatenate(test_labels)

# Parameters of LDA
lda_means, lda_covs = params(train_lda)

# Gaussian fit to every data-set
pred = normal_dist(test_lda, lda_means, lda_covs)

# Creating matrixes of 1 for every target for accuracy checks
target_classes = [i*np.ones(len(test[i])) for i in range(C)]

# Accuracy
accuracy = [np.sum(pred[test_labels == i] == i) / len(target_classes[i]) * 100 for i in range(C)]
overall_accuracy = np.sum(pred == test_labels)/len(test_labels) * 100
print('Accuracy per data-set: {0}'.format(accuracy))
print('Overall accuracy: {0}'.format(overall_accuracy))

# Confusion matrix plot
cnf = confusion_matrix(test_labels, pred)
cnf_m = ConfusionMatrixDisplay(cnf)
cnf_m.plot()
plt.show()

# For points plotting
'''
colors = ['red', 'green', 'blue', 'yellow', 'black', 'cyan', 'magenta', 'brown', 'orange', 'dodgerblue']
for i in range(C):
    data = np.asarray(train_lda[i]).T
    data2 = np.asarray(test_lda[i]).T
    plt.scatter(data[:,0], data[:,1], c=colors[i])
    #plt.scatter(data2[:,0], data2[:,1], c=colors[i], marker='x')
    plt.pause(1)
'''