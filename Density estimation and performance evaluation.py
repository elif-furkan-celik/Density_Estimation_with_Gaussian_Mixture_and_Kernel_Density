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
    gmm = GaussianMixture(n_components=i)
    gmm.fit(X_train)

    score = gmm.score_samples(X_test)

    y_pred = gmm.predict(X_test)
    print('Accuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
    plt.fill(X_test, np.exp(score), c='blue')
    plt.title("Gaussian {}".format(i))

    plt.show()

#Best Gaussian
gmm = GaussianMixture(n_components=3)
gmm.fit(X_train)
score = gmm.score_samples(X_test)

y_pred = gmm.predict(X_test)
print('Accuracy:{0:.3f}'.format(accuracy_score(y_test, y_pred)))
plt.fill(X_test, np.exp(score), c='blue')
plt.title("Gaussian 2")
plt.show()

"""#Kernel"""

kernels = ['tophat', 'gaussian']
bandwidths = [0.01, 0.05, 0.1, 0.5, 1, 2, 5]

for kern in kernels:
    for b in bandwidths:

        kde_skl = KernelDensity(kernel= kern, bandwidth=b)
        kde_skl.fit(X_train)

        score = kde_skl.score_samples(X_test)
        plt.fill(X_test, np.exp(score), c='green')
        plt.title("Kernel: {} and bandwidth: {}".format(kern, b))

        plt.show()


#Best KernelDensity
kde_skl = KernelDensity(kernel= 'gaussian', bandwidth=0.05)
kde_skl.fit(X_train)

score = kde_skl.score_samples(X_test)
plt.fill(X_test, np.exp(score), c='green')
plt.title("Kernel: gaussian and bandwidth: 0.05")
plt.show()