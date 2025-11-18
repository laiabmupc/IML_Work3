from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import matplotlib.pyplot as plt
import numpy as np

def agglomerative_clustering(df, config):
    '''
    Performs agglomerative clustering on the provided dataset.
    Returns the model labels and the built dendrogram.
    '''
    model = AgglomerativeClustering(n_clusters=config['n_clusters'],
                                    linkage=config['linkage'],
                                    metric=config['similarity_metric'],
                                    compute_distances=True).fit(df)

    dendrogram = plot_dendrogram(model, config['n_clusters'])
    return model.labels_, dendrogram


def plot_dendrogram(model, k):
    '''
    Create linkage matrix and then plot the dendrogram
    SOURCE CODE: sklearn
    '''
    # Create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack([model.children_, model.distances_, counts]).astype(float)

    # Plot
    dendrogram(linkage_matrix, truncate_mode='level', p=k, color_threshold=0)

    ax = plt.gca()
    for line in ax.get_lines():
        line.set_color("mediumvioletred")

    for coll in ax.collections:
        coll.set_color("mediumvioletred")

    return plt.gcf()