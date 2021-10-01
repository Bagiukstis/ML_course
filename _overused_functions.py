import numpy as np

class Overused():
    def params(self, data):
        '''
        :param data: pass a numpy array
        :return: returns covariance matrix and mean of the data
        '''
        cov = np.cov(data.T)  # 784 pivot entries
        mean = np.mean(data, axis=0)  # 784 dimensions
        return cov, mean