'''
MM13 exercise
A decision tree classifier and regressor based on Boston housing data.
We are predicting the price where the only parameter which is tuned is max_depth.
The cross-validation procedure finds the perfect max_depth fit which maximizes the result on the current data.
Future implementation might include: testing more parameters, adding aditional features, or using boosting.
'''

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import logging
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def randomforest_classification(X_train, y_train, X_test, y_test, max_depth=None, cross_validation=False):
    # Default max_depth value
    if max_depth == None:
        max_depth = 2

    # Randomforest classifier
    random_forest = RandomForestClassifier(max_depth=max_depth)
    random_forest.fit(X_train, y_train)
    predict = random_forest.predict(X_test)

    # Accuracy
    acc = np.sum(predict == y_test) / len(y_test) * 100
    logging.info('RandomForest pure accuracy: {0}'.format(acc))
    logging.info('Max_depth param: {0}'.format(max_depth))

    # Cross-validation option
    if cross_validation == True:
        random_forest = RandomForestClassifier()

        # Running through different max_depth options
        pipe = {
            'max_depth': np.arange(1,156),
        }
        gscv = GridSearchCV(random_forest, pipe)
        gscv.fit(X_train, y_train)
        gscv_predict = gscv.predict(X_test)

        # Accuracy of grid search
        acc_gscv = np.sum(gscv_predict == y_test) / len(y_test) * 100

        logging.info('RandomForest cross-validated accuracy: {0}'.format(acc_gscv))
        logging.info('Best parameters: {0}'.format(gscv.best_params_))
        return gscv_predict
    return predict
def randomforest_regression(X_train, y_train, X_test, y_test, max_depth, cross_validation=False):
    # Default max_depth value
    if max_depth == None:
        max_depth = 2

    # Randomforest classifier
    random_forest = RandomForestRegressor(max_depth=max_depth)
    random_forest.fit(X_train, y_train)
    predict = random_forest.predict(X_test)

    # Mean-squared-error
    mse = mean_squared_error(y_test, predict)
    logging.info('RandomForest pure MSE: {0}'.format(mse))
    logging.info('Max_depth param: {0}'.format(max_depth))

    # Cross-validation option
    if cross_validation == True:
        random_forest = RandomForestRegressor()

        # Running through different max_depth options
        pipe = {
            'max_depth': np.arange(1,150),
        }
        gscv = GridSearchCV(random_forest, pipe)
        gscv.fit(X_train, y_train)
        gscv_predict = gscv.predict(X_test)

        # Mean-squared-error of grid search
        mse_gscv = mean_squared_error(y_test, gscv_predict)

        logging.info('RandomForest cross-validated MSE: {0}'.format(mse_gscv))
        logging.info('Best parameters: {0}'.format(gscv.best_params_))
        return gscv_predict
    return predict

# Logger
logger = logging.getLogger()
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')
file_handler = logging.FileHandler('Logs/MM13.log')
file_handler.setLevel(logging.DEBUG)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# Data
raw_df = pd.read_csv('./MM13_material/BostonHousing.csv')
df = raw_df.drop(columns=['medv'])

# For regression
target_regression = raw_df['medv']

prices_raw = raw_df['medv'].values
prices_mean_raw = prices_raw.mean()

# For classification
container = np.zeros(prices_raw.size)
for i in range(prices_raw.size):
    if prices_raw[i] < prices_mean_raw:
        container[i] = 0
    else:
        container[i] = 1

# Regression (Price prediction)
X_train, X_test, y_train, y_test = train_test_split(df, target_regression, test_size=0.2, shuffle=False)
predict = randomforest_regression(X_train, y_train, X_test, y_test, 2, True)

predict_series = pd.Series(predict)
sns.lineplot(x=y_test.index, y=y_test)
sns.lineplot(x=y_test.index, y=predict_series, color='g')
plt.show()

'''
### Uncomment for classification

# Classification (low price, high price)
X_train, X_test, y_train, y_test = train_test_split(df, container, test_size=0.2, shuffle=False)
predict = randomforest_classification(X_train, y_train, X_test, y_test, 2, True)
cnf = confusion_matrix(y_test, predict)
cnf_m = ConfusionMatrixDisplay(cnf)
cnf_m.plot()
cnf_m.ax_.set(title='RandomForest classification prediction using best params')
plt.show()
'''


