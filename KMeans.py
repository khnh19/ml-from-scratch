import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from math import dist


class KMeans:
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        self.clusters = [[] for _ in range(K)]

        # mean for each cluster
        self.centroids = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize
        random_idxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroids = self.X[random_idxs]

        # main loop
        for _ in range(self.max_iters):
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                self.plot()

            # update centroids
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # check if converged
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                self.plot()

        # classify the samples
        return self._get_cluster_labels(self.clusters)

    def _get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it belongs to
        labels = np.empty(self.n_samples)

        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx

        return labels

    # create clusters
    def _create_clusters(self, centroids):
        clusters = [[] for _ in range(self.K)]

        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)

        return clusters

    # find closest centroid to sample
    def _closest_centroid(self, sample, centroids):
        distances = [dist(sample, p) for p in centroids]
        closest_idx = np.argmin(distances)
        return closest_idx

    # update centroids = mean of cluster
    def _get_centroids(self, clusters):
        centroids = np.zeros((self.K, self.n_features))

        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean

        return centroids

    # check if old and new centroids are the same
    def _is_converged(self, centroids_old, centroids):
        distances = [dist(centroids_old[i], centroids[i])
                     for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, idx in enumerate(self.clusters):
            p = self.X[idx].T
            ax.scatter(*p)

        for p in self.centroids:
            ax.scatter(*p, marker='x', color='black', linewidth=2)

        plt.show()


def main():
    np.random.seed(86)

    X, y = datasets.make_blobs(
        n_samples=500, n_features=2, centers=3, shuffle=True, random_state=40)

    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)
    k = KMeans(K=clusters, max_iters=150, plot_steps=True)
    
    y_kmeans = k.predict(X)
    k.plot()


if __name__ == '__main__':
    main()
