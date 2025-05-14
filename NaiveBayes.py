import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split


class NaiveBayes:
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self._classes = np.unique(y)
        n_classes = len(self._classes)

        # mean to calc PDF
        self._mean = np.zeros((n_classes, n_features), dtype=np.float64)
        # variance to calc PDF
        self._var = np.zeros((n_classes, n_features), dtype=np.float64)
        # prior probability
        self._prior = np.zeros(n_classes, dtype=np.float64)

        for idx, c in enumerate(self._classes):
            X_c = X[y == c]
            self._mean[idx, :] = X_c.mean(axis=0)
            self._var[idx, :] = X_c.var(axis=0)
            self._prior[idx] = X_c.shape[0] / float(n_samples)

    # probability density function
    def _pdf(self, class_idx, x):
        mean = self._mean[class_idx]
        var = self._var[class_idx]
        numerator = np.exp(-((x - mean) ** 2) / (2 * var))
        denominator = np.sqrt(2 * np.pi * var)

        return numerator / denominator

    # predict for each sample
    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    # predict for 1 sample
    def _predict(self, x):
        posteriors = []
        for idx, c in enumerate(self._classes):
            prior = np.log(self._prior[idx])

            # sum of log PDF
            class_cond = np.sum(np.log(self._pdf(idx, x)))

            posterior = prior + class_cond
            posteriors.append(posterior)

        return self._classes[np.argmax(posteriors)]


def main():
    X, y = datasets.make_classification(
        n_samples=1000, n_features=10, n_classes=2, random_state=12345
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=14
    )

    nb = NaiveBayes()
    nb.fit(X_train, y_train)
    predictions = nb.predict(X_test)

    acc = np.sum(predictions == y_test) / len(y_test)
    print(acc)


if __name__ == "__main__":
    main()
