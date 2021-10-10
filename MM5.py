from scipy.io import loadmat
import numpy as np
import matplotlib.pyplot as plt
import logging
from _overused_functions import Overused
from scipy.stats import multivariate_normal
from random import randrange

om = Overused()
#Setting up a logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

file_handler = logging.FileHandler('Logs/MM5.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)

logger.addHandler(file_handler)

def to_3_classes(data, train=True):
    '''
    :param data: raw data setimport matplotlib.pyplot as plt
    :param train: Boolean for a train set
    :return: either train classes or test classes
    '''
    if train:
        class_5 = data['trn5_2dim']
        class_6 = data['trn6_2dim']
        class_8 = data['trn8_2dim']
        return class_5, class_6, class_8
    class_5 = data['tst5_2dim']
    class_6 = data['tst6_2dim']
    class_8 = data['tst8_2dim']
    return class_5, class_6, class_8
def assign_label(class_1, xy_test):
    '''
    :param class_1: likelihood class label
    :param xy_test: real data
    :return: a list of points belonging to X distribution and Y distribution
    '''
    plot_list_1 = []
    plot_list_2 = []
    plot_list_3 = []
    xy_data = xy_test.tolist()
    for i in range(len(class_1)):
        if class_1[i] == 0:
            plot_list_1.append(xy_data[i])
        if class_1[i] == 1:
            plot_list_2.append(xy_data[i])
        if class_1[i] == 2:
            plot_list_3.append(xy_data[i])
    plt_numpy_1 = np.asarray(plot_list_1)
    plt_numpy_2 = np.asarray(plot_list_2)
    plt_numpy_3 = np.asarray(plot_list_3)
    return plt_numpy_1, plt_numpy_2, plt_numpy_3
def k_means(data):
    #find a function to make it random later
    r_5_prev = data[randrange(len(data))]
    r_6_prev = data[randrange(len(data))]
    r_8_prev = data[randrange(len(data))]
    label = np.zeros(len(data))
    iterations = 0
    while True:
        iterations += 1
        for i in range(len(c_2_dim)):
            distance_1 = np.linalg.norm(data[i] - r_5_prev)
            distance_2 = np.linalg.norm(data[i] - r_6_prev)
            distance_3 = np.linalg.norm(data[i] - r_8_prev)
            arg_min = np.argmin(np.c_[distance_1, distance_2, distance_3])
            label[i] = arg_min
        grp_1, grp_2, grp_3 = assign_label(label, data)
        r_5 = np.mean(grp_1, axis=0)
        r_6 = np.mean(grp_2, axis=0)
        r_8 = np.mean(grp_3, axis=0)

        mean_diff_1 = np.linalg.norm(r_5 - r_5_prev)
        mean_diff_2 = np.linalg.norm(r_6 - r_6_prev)
        mean_diff_3 = np.linalg.norm(r_8 - r_8_prev)

        if mean_diff_1 < 1 and mean_diff_2 < 1 and mean_diff_3 < 1:
            logger.info('Iterations required before convergence i = {0}'.format(iterations))
            break
        else:
            r_5_prev = r_5
            r_6_prev = r_6
            r_8_prev = r_8
    return grp_1, grp_2, grp_3
def EM_algorithm(cluster_1, cluster_2, cluster_3, data):
    iterations = 0
    norm_1_prev = 0
    norm_2_prev = 0
    norm_3_prev = 0
    while True:
        iterations += 1
        # Params for distributions
        cov_1, mean_1 = om.params(cluster_1)
        cov_2, mean_2 = om.params(cluster_2)
        cov_3, mean_3 = om.params(cluster_3)

        # Multi-dimensional distance tracking
        norm_1 = np.linalg.norm(cov_1)
        norm_2 = np.linalg.norm(cov_2)
        norm_3 = np.linalg.norm(cov_3)

        # Likelihoods (Component densities P(x|G_i))
        dist_1 = multivariate_normal(mean=mean_1, cov=cov_1)
        dist_2 = multivariate_normal(mean=mean_2, cov=cov_2)
        dist_3 = multivariate_normal(mean=mean_3, cov=cov_3)

        # Priors (Mixture proportions P(G_i)
        prior_1 = cluster_1.size / data.size
        prior_2 = cluster_2.size / data.size
        prior_3 = cluster_3.size / data.size

        # Posteriors P(x)
        post_1 = dist_1.pdf(data) * prior_1
        post_2 = dist_2.pdf(data) * prior_2
        post_3 = dist_3.pdf(data) * prior_3
        arg_min = np.argmax(np.c_[post_1, post_2, post_3], axis=1)
        cluster_1, cluster_2, cluster_3 = assign_label(arg_min, c_2_dim)

        if norm_1 - norm_1_prev < 1 and norm_2 - norm_2_prev < 1 and norm_3 - norm_3_prev < 1:
            n1 = np.linalg.norm(mean_1)
            n2 = np.linalg.norm(mean_2)
            n3 = np.linalg.norm(mean_3)
            means = [mean_1, mean_2, mean_3]
            covs = [cov_1, cov_2, cov_3]
            sort = np.argsort([n1, n2, n3])
            logger.info('Iterations required before convergence i = {0}'.format(iterations))
            for idx, val in enumerate(sort):
                if idx == 2:
                    logger.info('Cluster 6 mean: {0}, covariance: {1}'.format(means[val], covs[val]))
                if idx == 1:
                    logger.info('Cluster 8 mean: {0}, covariance: {1}'.format(means[val], covs[val]))
                if idx == 0:
                    logger.info('Cluster 5 mean: {0}, covariance: {1}'.format(means[val], covs[val]))
            break
        else:
            norm_1_prev = norm_1
            norm_2_prev = norm_2
            norm_3_prev = norm_3
    return cluster_1, cluster_2, cluster_3, arg_min

data = loadmat('MM5_material/2D568class.mat')
logging.critical('-----------------------------------------------------------------------------------------------------------')
c_5, c_6, c_8 = to_3_classes(data, train=True)

c_2_dim = np.concatenate([c_5, c_6, c_8])

grp_1, grp_2, grp_3 = k_means(c_2_dim)

em_1, em_2, em_3, pred = EM_algorithm(grp_1, grp_2, grp_3, c_2_dim)
mean_1 = np.mean(em_1, axis=1)
mean_2 = np.mean(em_2, axis=1)
mean_3 = np.mean(em_3, axis=1)

logging.shutdown()

########### PLOTTING ###############
plt.scatter(em_1[:,0], em_1[:,1], edgecolors='orange')
plt.scatter(mean_1[0], mean_1[1], marker='x')
plt.scatter(em_2[:,0], em_2[:,1], edgecolors='green')
plt.scatter(mean_2[0], mean_2[1], marker='x')
plt.scatter(em_3[:,0], em_3[:,1], edgecolors='blue')
plt.scatter(mean_3[0], mean_3[1], marker='x')
plt.show()

# plt.scatter(grp_1[:,0], grp_1[:,1], edgecolors='green')
# plt.scatter(mean_1[0], mean_1[1], marker='x')
# plt.scatter(grp_2[:,0], grp_2[:,1], edgecolors='blue')
# plt.scatter(mean_2[0], mean_2[1], marker='x')
# plt.scatter(grp_3[:,0], grp_3[:,1], edgecolors='orange')
# plt.scatter(mean_3[0], mean_3[1], marker='x')
# plt.show()
