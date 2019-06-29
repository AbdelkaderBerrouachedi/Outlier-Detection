from sklearn.linear_model import LogisticRegression

import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

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
test_data_1 = pd.read_csv('data/Synth/circles_moons_IGNG_outliers.csv')#pd.read_csv('data/Cardio/IGNG_outliers.csv')
test_data_2 = pd.read_csv('data/Synth/moons_circles_IGNG_outliers.csv')

test_data_3 = pd.read_csv('data/Synth/blobs_circles_IGNG_outliers.csv')
test_data_4 = pd.read_csv('data/Synth_ES/moons_circles_IGNG_outliers.csv')

test_data_5 = pd.read_csv('data/Synth/moons_blobs_IGNG_outliers.csv')
test_data_6 = pd.read_csv('data/Synth/blobs_moons_IGNG_outliers.csv')
test_data_7 = pd.read_csv('data/Synth/circles_moons_IGNG_outliers.csv')
test_data_8 = pd.read_csv('data/Synth/circles_blobs_IGNG_outliers.csv')


n_outliers = 200

train_data_1 = pd.read_csv('data/Synth/Train/0.csv_IGNG_outliers.csv')#pd.read_csv('data/Cardio/IGNG_outliers.csv')
train_data_1 = scaler.fit_transform(train_data_1)
train_data_1_input = pd.read_csv('data/Synth/Train/0.csv',header=None)
plt.scatter(x=train_data_1_input.iloc[:, 0], y=train_data_1_input.iloc[:, 1], s=10, c='b', marker="s", label='data')
plt.scatter(x=train_data_1_input.iloc[:n_outliers, 0], y=train_data_1_input.iloc[:n_outliers, 1], s=12, c='G', marker="s", label='outliers')
plt.show()

train_data_2 = pd.read_csv('data/Synth/Train/1.csv_IGNG_outliers.csv')
train_data_2 = scaler.fit_transform(train_data_2)
train_data_2_input = pd.read_csv('data/Synth/Train/1.csv',header=None)
plt.scatter(x=train_data_2_input.iloc[:, 0], y=train_data_2_input.iloc[:, 1], s=10, c='b', marker="s", label='data')
plt.scatter(x=train_data_2_input.iloc[:n_outliers, 0], y=train_data_2_input.iloc[:n_outliers, 1], s=12, c='G',
            marker="s", label='outliers')
plt.show()

train_data_3 = pd.read_csv('data/Synth/Train/2.csv_IGNG_outliers.csv')
train_data_3 = scaler.fit_transform(train_data_3)
train_data_3_input = pd.read_csv('data/Synth/Train/2.csv',header=None)
plt.scatter(x=train_data_3_input.iloc[:, 0], y=train_data_3_input.iloc[:, 1], s=10, c='b', marker="s", label='data')
plt.scatter(x=train_data_3_input.iloc[:n_outliers, 0], y=train_data_3_input.iloc[:n_outliers, 1], s=12, c='G', marker="s", label='outliers')
plt.show()


# train_data_4 = pd.read_csv('data/Synth/circles_blobs_IGNG_outliers.csv')
train_data_4 = pd.read_csv('data/Synth/Train/3.csv_IGNG_outliers.csv')
train_data_4 = scaler.fit_transform(train_data_4)
train_data_4_input = pd.read_csv('data/Synth/Train/3.csv',header=None)
plt.scatter(x=train_data_4_input.iloc[:, 0], y=train_data_4_input.iloc[:, 1], s=10, c='b', marker="s", label='data')
plt.scatter(x=train_data_4_input.iloc[:n_outliers, 0], y=train_data_4_input.iloc[:n_outliers, 1], s=12, c='G', marker="s", label='outliers')
plt.show()


train_data_5 = pd.read_csv('data/Synth/Train/4.csv_IGNG_outliers.csv')
train_data_5 = scaler.fit_transform(train_data_5)
train_data_5_input = pd.read_csv('data/Synth/Train/4.csv',header=None)
# plt.scatter(train_data_5_input.iloc[:,0],train_data_5_input.iloc[:,1])
# plt.show()



train_data = np.concatenate((train_data_1, train_data_2, train_data_3, train_data_4), axis=0)
# train_data2 = pd.read_csv('data/Synth/Moons/blobs_IGNG_outliers.csv', usecols=['min_distances','avg_k_distances','max_k_distances',
#                                                                 'outlier_K_factor','lof_clusters','cluster_sparsity','label'])#pd.read_csv('data/ALOI/IGNG_outliers.csv')
# train_data3 = pd.read_csv('data/Cardio/IGNG_outliers.csv', usecols=['min_distances','avg_k_distances','max_k_distances',
#                                                                 'outlier_K_factor','lof_clusters','cluster_sparsity','label'])#pd.r
# test_data_input = pd.read_csv('data/Synth/circles_blobs_input_data.csv', header=None)
# labels_data = np.concatenate([train_data.iloc[:, -1].values, test_data.iloc[:, -1].values])
# print(train_data.shape, test_data.shape, test_data_input.shape)
# train_data = train_data_6
test_data_input = train_data_4_input#pd.read_csv('data/Satellite/satellite.csv', header=None)

test_data = train_data_4 #pd.read_csv('data/Satellite/IGNG_outliers.csv', usecols=['min_distances','avg_k_distances','max_k_distances',
                                                               # 'outlier_K_factor','lof_clusters','cluster_sparsity','label'])

# columns = list(test_data.columns)

# train_data = pd.concat((train_data1, train_data2, train_data3),axis=0)
# test_data = pd.read_csv('data/ALOI/IGNG_outliers.csv', usecols=['min_distances','avg_k_distances','max_k_distances',
#                                                                 'outlier_K_factor','lof_clusters','cluster_sparsity','label'])#pd.re

# train_data_AE = pd.read_csv('data/Cardio/AE_outliers.csv')
# test_data_AE = pd.read_csv('data/ALOI/AE_outliers.csv')
# train_data_AE = scaler.fit_transform(train_data_AE.iloc[:, 1:-1])
# test_data_AE = scaler.fit_transform(test_data_AE.iloc[:, 1:-1])

train_data_np = scaler.fit_transform(train_data)
test_data_np = scaler.fit_transform(test_data)
# combined_data_np = np.concatenate([train_data_np, test_data_np], axis=0)
# combined_data_np = np.concatenate([combined_data_np, labels_data.reshape(-1, 1)], axis=1)

# combined_df = pd.DataFrame(data=combined_data_np, columns=columns)

# train_data, test_data = train_test_split(shuffle(combined_df.values), test_size=0.25)

X_train_data = train_data_np[:, :-1]
y_train_data = train_data_np[:, -1]
X_test_data = test_data_np[:, :-1]
y_test_data = test_data_np[:, -1]
# print(X_train_data.shape, y_train_data.shape, X_test_data.shape, y_test_data.shape)
clf = LogisticRegression(random_state=0, solver='lbfgs').fit(X_train_data, y_train_data)
print('*** TRAIN ***')
compute_metrics(clf, X_train_data, y_train_data)
print('*** TEST ***')
compute_metrics(clf, X_test_data, y_test_data)

# print('******** With AE **********')
# train_data_ae = np.concatenate((train_data[:, :-1], train_data_AE), axis=1)
# test_data_ae = np.concatenate((test_data[:, :-1], test_data_AE), axis=1)
# clf = LogisticRegression(random_state=0, solver='lbfgs').fit(train_data_ae, train_data[:, -1])
# print('*** TRAIN ***')
# compute_metrics(clf, train_data_ae, train_data[:, -1])
# print('*** TEST ***')
# compute_metrics(clf, test_data_ae, test_data[:, -1])

# print('******** SVM **********')
# clf = SVC(random_state=0, C=1, gamma='scale').fit(X_train_data, y_train_data)
# print('*** TRAIN ***')
# compute_metrics(clf, X_train_data, y_train_data)
# print('*** TEST ***')
# compute_metrics(clf, X_test_data, y_test_data)

# train_score = clf.score(train_data[:, :-1], train_data[:, -1])
# test_score = clf.score(test_data[:, :-1], test_data[:, -1])
# print("Mean train accuracy: {:.4f} , Mean test accuracy: {:.4f}".format(train_score, test_score))
y_pred_test = clf.predict(X_test_data)
df_result = pd.DataFrame(data=np.concatenate((y_pred_test.reshape(-1, 1), y_test_data.reshape(-1, 1)),axis=1), columns=['pred', 'actual'])
recon_outliers = test_data_input[y_pred_test == 1]
fig, ax = plt.subplots()
ax.scatter(x=test_data_input.iloc[:, 0], y=test_data_input.iloc[:, 1], s=10, c='b', marker="s", label='data')
ax.scatter(x=test_data_input.iloc[:n_outliers, 0], y=test_data_input.iloc[:n_outliers, 1], s=12, c='G', marker="s", label='outliers')
ax.scatter(x=recon_outliers.iloc[:, 0], y=recon_outliers.iloc[:, 1], s=6, c='r', marker="o", label='predicted')
plt.show()


print('finish')

