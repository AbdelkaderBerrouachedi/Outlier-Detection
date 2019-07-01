from sklearn import datasets
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons, make_blobs

n_samples = 2000
outliers_fraction = 0.1
n_outliers = int(outliers_fraction * n_samples)

n_inliers = n_samples - n_outliers
rng = np.random.RandomState(42)
# Define datasets
blobs_params = dict(random_state=0, n_samples=n_inliers, n_features=2)
datasets = [
    make_blobs(centers=[[0, 0], [0, 0]], cluster_std=0.5,
               **blobs_params)[0],
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[0.5, 0.5],
               **blobs_params)[0],
    make_blobs(centers=[[2, 2], [-2, -2]], cluster_std=[1.5, .3],
               **blobs_params)[0],
    4. * (make_moons(n_samples=n_samples, noise=.05, random_state=0)[0] -
          np.array([0.5, 0.25])),
    14. * (np.random.RandomState(42).rand(n_samples, 2) - 0.5)]

for i_dataset, X in enumerate(datasets):

    X = np.concatenate((rng.uniform(low=-6, high=6, size=(n_outliers, 2)),  X[n_outliers:, :]), axis=0)
    labels = []
    labels = np.zeros(X.shape[0])
    for i in range(n_outliers):
        labels[i] = 1
    X_target = np.concatenate((X, labels.reshape(-1, 1)), axis=1)
    fig, ax = plt.subplots()
    df = pd.DataFrame(data=X_target, columns=['X', 'Y', 'target'])
    ax.scatter(x=df.iloc[:n_outliers, 0], y=df.iloc[:n_outliers, 1], s=10, c='b', marker="s", label='data')
    ax.scatter(x=df.iloc[n_outliers:, 0], y=df.iloc[n_outliers:, 1], s=10, c='r', marker="s", label='data')

    plt.show()
    df.to_csv('data/Synth/Train_STD/'+str(i_dataset)+'.csv', index=None, header= None)