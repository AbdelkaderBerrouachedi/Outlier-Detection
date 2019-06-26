from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import train_test_split

from sklearn.metrics import precision_score, accuracy_score,recall_score,f1_score,confusion_matrix,roc_auc_score,classification_report

from sklearn import preprocessing
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import os


def compute_metrics(model, X, y):
    y_pred = model.predict(X)
    precisionM = precision_score(y, y_pred, average='macro')
    precisionB = precision_score(y, y_pred, average='binary')
    precisionW = precision_score(y, y_pred, average='weighted')
    print('Precision M,W,B', precisionM, precisionW, precisionB)
    print('Accuracy', accuracy_score(y, y_pred))
    print('AUC', roc_auc_score(y, y_pred, average='macro'))
    print(classification_report(y, y_pred))

scaler = preprocessing.MinMaxScaler()
cardio_data = pd.read_csv('data/Cardio/IGNG_outliers.csv')
satellite_data = pd.read_csv('data/Satellite/IGNG_outliers.csv')
labels_data = np.concatenate([cardio_data.iloc[:, -1].values, satellite_data.iloc[:, -1].values])
columns = list(cardio_data.columns[1:])

cardio_data_np = scaler.fit_transform(cardio_data.iloc[:, 1:])
satellite_data_np = scaler.fit_transform(satellite_data.iloc[:, 1:])
# combined_data_np = np.concatenate([cardio_data_np, satellite_data_np], axis=0)
# combined_data_np = np.concatenate([combined_data_np, labels_data.reshape(-1, 1)], axis=1)

# combined_df = pd.DataFrame(data=combined_data_np, columns=columns)

# train_data, test_data = train_test_split(shuffle(combined_df.values), test_size=0.25)

train_data = cardio_data_np
test_data = satellite_data_np
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(train_data[:, :-1], train_data[:, -1])
print('*** TRAIN ***')
compute_metrics(clf, train_data[:, :-1], train_data[:, -1])
print('*** TEST ***')
compute_metrics(clf, test_data[:, :-1], test_data[:, -1])

# print('******** SVM **********')
# clf = SVC(random_state=0, C=1).fit(train_data[:, :-1], train_data[:, -1])
# print('*** TRAIN ***')
# compute_metrics(clf, train_data[:, :-1], train_data[:, -1])
# print('*** TEST ***')
# compute_metrics(clf, test_data[:, :-1], test_data[:, -1])
#
# train_score = clf.score(train_data[:, :-1], train_data[:, -1])
# test_score = clf.score(test_data[:, :-1], test_data[:, -1])
# print("Mean train accuracy: {:.4f} , Mean test accuracy: {:.4f}".format(train_score, test_score))
print('finish')