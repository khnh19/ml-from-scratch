import numpy as np
from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from math import dist
from collections import Counter


class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        predictions = [self._predict(x) for x in X]
        return predictions

    def _predict(self, x):
        distances = [dist(x, x_train) for x_train in self.X_train]

        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]

        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]


def main():
    X, y = load_iris(return_X_y=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=86)

    plt.figure()
    plt.scatter(X[:, 2], X[:, 3], c=y, cmap=ListedColormap(
        ['#FF0000', '#00FF00', '#0000FF']), edgecolor='k', s=20)
    plt.show()

    clf = KNN(k=3)
    clf.fit(X_train, y_train)
    predictions = clf.predict(X_test)

    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)


if __name__ == '__main__':
    main()
