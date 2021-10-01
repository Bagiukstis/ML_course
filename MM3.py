'''
The aim of this exercise is to classify points to either one multivariate distribution or the other.
The program evaluates the probability of one distribution and the other distribution and then decides
to which class do the points belong to.

Method used: MLE and MAP. Considering if we know something prior about our data or not.
'''

import pandas as pd
import numpy as np
from scipy.stats import multivariate_normal
import matplotlib.pyplot as plt
from _overused_functions import Overused

def load_np(df):
    numpy = np.asarray(df.values)
    return numpy
def plot_points(numpy):
    x, y = numpy.T
    return x, y
def params(numpy):
    mean = np.mean(numpy, axis=0)
    std = np.std(numpy)
    var = np.var(numpy)
    cov = np.cov(numpy.T)
    return mean, std, var, cov
def compare_classes(x_pdf, y_pdf):
    class_1 = []
    x_pdf_list = x_pdf.tolist()
    y_pdf_list = y_pdf.tolist()

    for i in range(len(x_pdf_list)):
        if x_pdf_list[i] > y_pdf_list[i]:
            class_1.append('1')
        else:
            class_1.append('2')
    return class_1
def compare_for_plotting(class_1, xy_test):
    '''
    :param class_1: likelihood class label
    :param xy_test: real data
    :return: a list of points belonging to X distribution and Y distribution
    '''
    plot_list_1 = []
    plot_list_2 = []
    xy_data = xy_test.tolist()
    for i in range(len(class_1)):
        if class_1[i] == '1':
            plot_list_1.append(xy_data[i])
        else:
            plot_list_2.append(xy_data[i])
    plt_numpy_1 = np.asarray(plot_list_1)
    plt_numpy_2 = np.asarray(plot_list_2)
    return plt_numpy_1, plt_numpy_2
def accuracy(class_comparison, test_class):
    class_np = np.asarray(class_comparison).T
    gg = np.subtract(class_np.astype(int), test_class.T)
    true_negative = np.sum(abs(gg))
    true_positive = class_np.size - true_negative
    accuracy = true_positive / class_np.size
    return accuracy
def rate(x_train, y_train):
    sum = len(x_train) + len(y_train)
    rate_x = len(x_train) / sum
    rate_y = 1 - rate_x
    return rate_x, rate_y

######Loading the datasets######
df = pd.read_csv('MM3_material/trn_x.txt', sep='  ', names=['col1', 'col2'])
df_1 = pd.read_csv('MM3_material/trn_y.txt', sep='  ', names=['col1', 'col2'])
df_3 = pd.read_csv('MM3_material/tst_xy.txt', sep='  ', names=['col1', 'col2'])
df_4 = pd.read_csv('MM3_material/tst_xy_class.txt', sep='  ', names=['col1'])
df_5 = pd.read_csv('MM3_material/tst_xy_126.txt', sep='  ', names=['col1', 'col2'])
df_6 = pd.read_csv('MM3_material/tst_xy_126_class.txt', sep='  ', names=['col1'])

##### Loading DF's as np arrays ####
x_train = load_np(df)
y_train = load_np(df_1)
xy_test = load_np(df_3)
xy_test_class = load_np(df_4)
xy_126_test = load_np(df_5)
xy_126_class = load_np(df_6)

##### Estimating parameters for distributions #####
cov_x, mean_x = Overused().params(x_train)
cov_y, mean_y = Overused().params(y_train)

##### Creating the distributions #####
l_x = multivariate_normal(mean=mean_x, cov = cov_x) #multivariate distribution for X (consists of X and Y points)
l_y = multivariate_normal(mean=mean_y, cov = cov_y) #second multivariate distribution for Y (consists of X and Y points)

x_rate, y_rate = rate(x_train, y_train) #calculating the dataset rate

''' 
Posterior density formula when prior is given:
P(theta|X) = (P(X|theta) * P(theta)) / P(X)
theta_MAP = argmax_theta(P(theta|x))

Posteriori density formula when no prior is given:
P(theta|X) = P(X|theta)
where theta is defined as theta_ML = argmax_theta(P(X|theta))
as no prior is given.
Thus parameters are set up by solving the ML equation for multivariate distribution for N random variables.

multivariate_normal.pdf() function solves for ML, thus if any prior is given - then we can multiply the probability density
function with given prior probabilities
'''

#Exercise 1
#Maximum a posteriori (MAP) as we know the length of both data sets and we can find the rate
x_pdf = l_x.pdf(xy_test) * x_rate
y_pdf = l_y.pdf(xy_test) * y_rate

class_1 = compare_classes(x_pdf, y_pdf)
model_accuracy = accuracy(class_1, xy_test_class)

#Exercise 2
#Maximum Likelihood (ML), as no prior probabilities are given.
#We can ignore the posterior, or uniformly distribute by 1/2 and 1/2. The result does not change.
x_pdf_2 = l_x.pdf(xy_126_test)
y_pdf_2 = l_y.pdf(xy_126_test)

class_2 = compare_classes(x_pdf_2, y_pdf_2)
model_accuracy_2 = accuracy(class_2, xy_126_class)

#Exercise 3
#Maximum a posteriori (MAP) assuming prior probability for X = 0.9 and y = 0.1
x_pdf_3 = l_x.pdf(xy_126_test) * 0.9
y_pdf_3 = l_y.pdf(xy_126_test) * 0.1

class_3 = compare_classes(x_pdf_3, y_pdf_3)
model_accuracy_3 = accuracy(class_3, xy_126_class)
print('Accuracy difference: {0} %'.format((model_accuracy_3-model_accuracy_2)*100))

######### FOR PLOTTING ############
#change the class argument for plotting comparison (class_1, class_2, class_3)
x_classified, y_classified = compare_for_plotting(class_1,xy_test)
x_1, x_2 = plot_points(x_train)
y_1, y_2 = plot_points(y_train)
x_c1, x_c2 = plot_points(x_classified)
y_c1, y_c2 = plot_points(y_classified)

plt.scatter(x_1, x_2, edgecolors='blue')
plt.scatter(y_1, y_2, edgecolors='green')
plt.scatter(x_c1, x_c2, edgecolors='red')
#plt.scatter(y_c1, y_c2, edgecolors='yellow')  # Has outliers
plt.show()


