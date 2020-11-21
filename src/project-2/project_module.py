# TODO: name this something more appropriate
import matplotlib.pyplot as plt
import matplotlib.cm as cm

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans



def kmeans_cluster_data(X, k=2, plot=True, fig=None, ax=None, seed=None):
    """Takes in a dataset and clusters it for the n dimensions of the dataset.
    If plot is enabled, the dataset is then reduced using pca before being plotted in a 2 dimensional space
    Dimensions are kept for the clusterer

    returns labels"""
    if seed is None:
        seed = 42
    clusterer = KMeans(n_clusters=k, random_state=seed)
    labels = clusterer.fit_predict(X)

    if plot:
        if not (fig or ax):
            fig = plt.figure(figsize=(12, 6))
            ax = fig.add_subplot()

        elif fig:
            ax = fig.add_subplot()

        colors = cm.nipy_spectral(labels.astype(float) / k)

        ax.set_title(f"{k}-means clustering")
        pca = PCA(n_components=2)

        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")

        xp = pca.fit_transform(X)
        centroids = pca.transform(clusterer.cluster_centers_)
        ax.scatter(*xp.T, s=30, c=colors, alpha=0.5)
        ax.scatter(*centroids.T, marker='o', c="white", alpha=1, s=200, edgecolor='k')
        for i, c in enumerate(centroids):
            ax.scatter(c[0], c[1], marker='$%d$' % i, alpha=1, s=50, edgecolor='k')

    return labels, clusterer
