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

Y = main_df.loc[:,'class']
# Y
# Separate the dataframe for input(X) and output variables(y)
# merged_dataframe = merged_dataframe.dropna()
X = main_df.loc[:,['rsi','sma','lma','adx','macd','signal','Open','Close','us_china','trump','malaysia','Score']]
# X = main_df.loc[:,['rsi','sma','lma','adx','macd','signal','us_china','trump','malaysia','Score']]

# y = main_df.loc[:,'class']
# Set the validation size, i.e the test set to 20%
# Split the dataset to test and train sets
# Split the initial 70% of the data as training set and the remaining 30% data as the testing set


split_percentage = 0.7
split = int(split_percentage*len(main_df))

X_train = X[:split]
Y_train = Y[:split]

X_test = X[split:]
Y_test = Y[split:]


# The best model = SVM,
svm = SVC().fit(X_train, Y_train)
accuracy_train = accuracy_score(Y_train, svm.predict(X_train))
accuracy_test = accuracy_score(Y_test, svm.predict(X_test))

print('Train accuracy   %.2f' %accuracy_train)
print('Test accuracy   %.2f' %accuracy_test)

main_df['Real_Returns'] = np.log(main_df['Close']/main_df['Close'].shift(1))
main_df['Predicted_Signal'] = svm.predict(X)
Cum_Real_Returns = main_df[split:]['Real_Returns'].cumsum()*100

main_df['Strategy_Returns'] = main_df['Real_Returns'] * main_df['Predicted_Signal'].shift(1)
Cum_Strategy_Returns = main_df[split:]['Strategy_Returns'].cumsum()*100

plt.figure(figsize=(10,5))
plt.plot(Cum_Real_Returns, color='r', label='Real Returns')
plt.plot(Cum_Strategy_Returns, color='g', label='Strategy Returns')
plt.legend()
plt.show()

main_df['Predicted_Signal'] = svm.predict(X)
# Calculate log returns
main_df['Return'] = np.log(main_df.Close.shift(-1) / main_df.Close)*100
main_df['Strategy_Return'] = main_df.Return * main_df.Predicted_Signal
main_df.Strategy_Return.iloc[split:].cumsum().plot(figsize=(10,5))
plt.ylabel("Strategy Returns (%)")
plt.show()


from sklearn.decomposition import PCA

pca = PCA(n_components=3)
pca.fit(X)
transformed = pca.transform(X)

transformed.shape
print(type(transformed))

# Dump components relations with features:
print pd.DataFrame(pca.components_,columns=X.columns,index = ['PC-1','PC-2','PC-3'])

pca_df = pd.DataFrame(transformed)
train_size = int(len(X.index) * 0.7)
X_train_pca, X_test_pca = pca_df.loc[0:train_size, :], pca_df.loc[train_size: len(X.index), :]

clf = XGBClassifier(n_estimators=500, max_depth=3)
clf.fit(X_train_pca, y_train)
y_pred_pca = clf.predict(X_test_pca)
score = accuracy_score(y_test, y_pred_pca)
print("Score is "+ str(score))