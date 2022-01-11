# Nice cock rights

import numpy as np
from sklearn.utils import shuffle

class Overused():
    def params(self, data):
        '''
        :param data: pass a numpy array
        :return: returns covariance matrix and mean of the data
        '''
        cov = np.cov(data.T)  # 784 pivot entries
        mean = np.mean(data, axis=0)  # 784 dimensions
        return cov, mean

    def to_sep(self, data, to_shuffle=True):
        x_train = []
        x_test = []
        for i in range(10):
            x_train.append(data['train%d' % i])
            x_test.append(data['test%d' % i])

        accuracy_target_classes = [i * np.ones(len(x_test[i])) for i in range(10)]

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