from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.mixture import GaussianMixture
import matplotlib.pyplot as plt
from sklearn.neighbors import KernelDensity
import numpy as np

data = load_iris()
data.feature_names, data.target_names


X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.33, random_state=3)
X_train.shape, X_test.shape, y_train.shape, y_test.shape

"""#Gaussian"""

from sklearn.metrics import accuracy_score  

for i in range(1,10):
    
    x = X_train[:,0]
    y = X_train[:,1]

    xy_train  = np.vstack([y, x]).T

    xx, yy = np.mgrid[x.min():x.max():100j, 
                      y.min():y.max():100j]

    xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T

    gmm = GaussianMixture(n_components=i)
    gmm.fit(xy_train)

    z = np.exp(gmm.score_samples(xy_sample))
    zz = np.reshape(z, xx.shape)
    print("Gaussian {}".format(i))
    y_pred = gmm.predict(X_test[:,:2])
    print('Accuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
    plt.pcolormesh(xx, yy, zz)
    plt.scatter(X_test[:,0], X_test[:,1], s=1, facecolor='white')
    plt.show()

"""#Kernel"""

kernels = ['tophat', 'gaussian']
bandwidths = [0.01, 0.05, 0.1, 0.5, 1, 4, 5]

for kern in kernels:
    for b in bandwidths:
        x = X_train[:,0]
        y = X_train[:,1]

        xy_train  = np.vstack([y, x]).T

        xx, yy = np.mgrid[x.min():x.max():100j, 
                          y.min():y.max():100j]

        xy_sample = np.vstack([yy.ravel(), xx.ravel()]).T

        kde_skl = KernelDensity(kernel= kern, bandwidth=b)
        kde_skl.fit(xy_train)

        z = np.exp(kde_skl.score_samples(xy_sample))
        zz = np.reshape(z, xx.shape)
        print("Kernel: {} and bandwidth: {}".format(kern, b))
        plt.pcolormesh(xx, yy, zz)
        plt.scatter(X_test[:,0], X_test[:,1], s=1, facecolor='white')
        plt.show()