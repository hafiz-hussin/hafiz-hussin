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
import matplotlib.pyplot as plt
import random


main_df = pd.read_csv('data_for_modelling.csv')
print(main_df.shape)
print(main_df.dtypes)

Y = np.where(main_df['class']==1,1,-1)
Y
# Separate the dataframe for input(X) and output variables(y)
# merged_dataframe = merged_dataframe.dropna()
X = main_df.loc[:,['rsi','sma','lma','adx','macd','signal','Open','Close','us_china','trump','malaysia','Score']]
# y = main_df.loc[:,'class']
# Set the validation size, i.e the test set to 20%
validation_size = 0.20
# Split the dataset to test and train sets
# Split the initial 70% of the data as training set and the remaining 30% data as the testing set
train_size = int(len(X.index) * 0.7)
print(len(y))
print(train_size)


split_percentage = 0.7
split = int(split_percentage*len(main_df))

X_train = X[:split]
Y_train = Y[:split]

X_test = X[split:]
Y_test = Y[split:]

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, Y_train)
accuracy_train = accuracy_score(Y_train, knn.predict(X_train))
accuracy_test = accuracy_score(Y_test, knn.predict(X_test))

print('Train accuracy   %.2f' %accuracy_train)
print('Test accuracy   %.2f' %accuracy_test)



main_df['Predicted_Signal'] = knn.predict(X)
main_df['Return'] = np.log(main_df.Close.shift(-1) / main_df.Close)*100
main_df['Strategy_Return'] = main_df.Return * main_df.Predicted_Signal


main_df.Strategy_Return.iloc[split:].cumsum().plot(figsize=(10,5))
plt.ylabel("Strategy Returns (%)")
plt.show()

# The best model = SVM,
svm = SVC().fit(X_train, Y_train)
accuracy_train = accuracy_score(Y_train, svm.predict(X_train))
accuracy_test = accuracy_score(Y_test, svm.predict(X_test))

print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))
print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))