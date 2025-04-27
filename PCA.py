import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris


class PCA:
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None

    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance
        cov = np.cov(X.T)

        # eigen decomposition
        eigvecs, eigvals = np.linalg.eig(cov)

        # transpose eigenvectors
        eigvecs = eigvecs.T

        # sort
        idxs = np.argsort(eigvals)[::-1]
        eigvecs = eigvecs[idxs]
        eigvals = eigvals[idxs]

        # select components
        self.components = eigvecs[:self.n_components]

    def transform(self, X):
        X = X - self.mean
        return np.dot(X, self.components.T)


def main():
    X, y = load_iris(return_X_y=True)

    pca = PCA(n_components=2)
    pca.fit(X)
    X_pca = pca.transform(X)

    print(f'shape of X: {X.shape}')
    print(f'shape of X_pca: {X_pca.shape}')

    x1 = X_pca[:, 0]
    x2 = X_pca[:, 1]

    plt.scatter(x1, x2, c=y, edgecolor='none', alpha=0.8,
                cmap=plt.cm.get_cmap('viridis', 3))

    plt.xlabel('principal component 1')
    plt.ylabel('principal component 2')
    plt.colorbar()
    plt.show()


if __name__ == '__main__':
    main()
