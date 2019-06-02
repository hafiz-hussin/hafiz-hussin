import pandas as pd
import numpy as np
import warnings
from matplotlib import pyplot
from pandas import Series, datetime
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score, GridSearchCV, TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import roc_curve, auc
import random
import matplotlib.pyplot as plt
from xgboost import XGBClassifier

main_df = pd.read_csv('data_for_modelling.csv')
print(main_df.shape)
print(main_df.dtypes)

X = main_df.loc[:,['rsi','sma','lma','adx','macd','signal','us_china','trump','malaysia','Score']]
# X = main_df.loc[:,['rsi','sma','lma','adx','macd','signal','Open','Close','us_china','trump','malaysia','Score']]
y = main_df.loc[:,'class']
# Set the validation size, i.e the test set to 20%
validation_size = 0.20
# Split the dataset to test and train sets
# Split the initial 70% of the data as training set and the remaining 30% data as the testing set
train_size = int(len(X.index) * 0.7)
print(len(y))
print(train_size)

X_train, X_test = X.loc[0:train_size, :], X.loc[train_size: len(X.index), :]
y_train, y_test = y[0:train_size+1], y.loc[train_size: len(X.index)]
print('Observations: %d' % (len(X.index)))
print('X Training Observations: %d' % (len(X_train.index)))
print('X Testing Observations: %d' % (len(X_test.index)))
print('y Training Observations: %d' % (len(y_train)))
print('y Testing Observations: %d' % (len(y_test)))


# MOdels
# num_folds = 10
# scoring = 'accuracy'
# Append the models to the models list
models = []
models.append(('LR' , LogisticRegression()))
models.append(('LDA' , LinearDiscriminantAnalysis()))
models.append(('KNN' , KNeighborsClassifier()))
models.append(('CART' , DecisionTreeClassifier()))
models.append(('NB' , GaussianNB()))
models.append(('SVM' , SVC()))
models.append(('RF' , RandomForestClassifier(n_estimators=50)))
models.append(('XGBoost', XGBClassifier()))

# Evaluate each algorithm for accuracy
results = []
names = []
for name, model in models:
    clf = model
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy_train = accuracy_score(y_train, clf.predict(X_train))
    accuracy_test = accuracy_score(y_test, clf.predict(X_test))
    # print(name + ': Train Accuracy:{: .2f}%'.format(accuracy_train*100))
    print(name +': Test Accuracy:{: .2f}%'.format(accuracy_test*100))

# The best model = SVM,
cls = SVC().fit(X_train, y_train)
accuracy_train = accuracy_score(y_train, cls.predict(X_train))
accuracy_test = accuracy_score(y_test, cls.predict(X_test))

print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))
print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))

split_percentage = 0.7
split = int(split_percentage*len(main_df))

main_df['Predicted_Signal'] = cls.predict(X)
main_df['Return'] = np.log(main_df.Close.shift(-1) / main_df.Close)*100
main_df['Strategy_Return'] = main_df.Return * main_df.Predicted_Signal
#
#
main_df.Strategy_Return.plot(figsize=(10,5))
plt.ylabel("Strategy Returns (%)")
plt.show()


main_df['Real_Returns'] = np.log(main_df['Close']/main_df['Close'].shift(1))
main_df['Predicted_Signal'] = cls.predict(X)
Cum_Real_Returns = main_df[split:]['Real_Returns'].cumsum()*100

main_df['Strategy_Returns'] = main_df['Real_Returns'] * main_df['Predicted_Signal'].shift(1)
Cum_Strategy_Returns = main_df[split:]['Strategy_Returns'].cumsum()*100

plt.figure(figsize=(10,5))
plt.plot(Cum_Real_Returns, color='r', label='Real Returns')
plt.plot(Cum_Strategy_Returns, color='g', label='Strategy Returns')
plt.legend()
plt.show()

# from sklearn.decomposition import PCA
#
# pca = PCA(n_components=3)
# pca.fit(X)
# transformed = pca.transform(X)
#
# transformed.shape
# print(type(transformed))
#
# # Dump components relations with features:
# print pd.DataFrame(pca.components_,columns=X.columns,index = ['PC-1','PC-2','PC-3'])
#
# pca_df = pd.DataFrame(transformed)
# train_size = int(len(X.index) * 0.7)
# X_train_pca, X_test_pca = pca_df.loc[0:train_size, :], pca_df.loc[train_size: len(X.index), :]
#
# clf = XGBClassifier(n_estimators=500, max_depth=3)
# clf.fit(X_train_pca, y_train)
# y_pred_pca = clf.predict(X_test_pca)
# score = accuracy_score(y_test, y_pred_pca)
# print("Score is "+ str(score))


std = Cum_Strategy_Returns.std()
sharpe = (Cum_Strategy_Returns-Cum_Real_Returns)/std
sharpe = sharpe.mean()
print('Sharpe Ratio: %.2f' %sharpe)