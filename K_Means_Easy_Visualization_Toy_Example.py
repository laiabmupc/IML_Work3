import numpy as np
import pandas as pd
from K_Means import *
from K_Means_pp import *
from FEKM import *

from sklearn.metrics import confusion_matrix

import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.preprocessing import MinMaxScaler
import random

#################################
## CREATE DATASET & VISUALIZE ##
#################################

def create_toy_dataset():
    """Creates a 2D dataset normalized between 0 and 1."""
    # 1. Create clearly separated blobs
    X_raw, y_true = make_blobs(n_samples=2000,
                               centers=3,
                               n_features=2,
                               cluster_std=1.5)  # Creates distinct clusters

    # 2. Normalize the data to be between 0 and 1
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X_raw)

    print(f"Dataset shape: {X_normalized.shape}")
    print(f"Dataset min: {np.min(X_normalized)}, max: {np.max(X_normalized)}")
    return X_normalized


def plot_clusters(final_clusters):
    """Plots the final clusters and their centroids."""
    plt.figure(figsize=(10, 8))

    # Use a nice color list
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

    # 1. Plot the points for each cluster
    for centroid_id, data in final_clusters.items():
        points_list = data['points']

        if len(points_list) > 0:
            # Convert list of points to a NumPy array for plotting
            points_array = np.array(points_list)

            plt.scatter(points_array[:, 0],  # x-values
                        points_array[:, 1],  # y-values
                        color=colors[centroid_id % len(colors)],
                        label=f'Cluster {centroid_id}',
                        alpha=0.7)  # Add transparency

    # 2. Plot the final centroids on top
    centroid_centers = []
    for centroid_id, data in final_clusters.items():
        center = data['center']
        centroid_centers.append(center)

    centroid_array = np.array(centroid_centers)
    plt.scatter(centroid_array[:, 0], centroid_array[:, 1],
                marker='X',  # Use a big 'X'
                s=250,  # Make it large
                color='black',  # Make it black
                edgecolors='white',  # White border for visibility
                linewidth=1.5,
                label='Final Centroids',
                zorder=10)  # Ensure centroids are on top

    plt.title('K-Means Clustering Results (From Scratch)', fontsize=16)
    plt.xlabel('Feature 1 (Normalized)', fontsize=12)
    plt.ylabel('Feature 2 (Normalized)', fontsize=12)
    plt.legend(fontsize=10)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.show()

##########
## MAIN ##
##########

if __name__ == "__main__":
        # 1. Create the normalized toy dataset
        print("Creating toy dataset...")
        X_data = create_toy_dataset()

        # 2. Create and fit the K-Means model
        k = 3  # We know the toy dataset has 3 centers
        print(f"Running K-Means with k={k}...")
        kmeans_model = FEKM(k=k, X=X_data)

        # The plot_clusters function uses the first return value
        final_clusters_data, _ = kmeans_model.fit(max_iterations=100, tolerance=0)
        print("K-Means fitting complete.")

        # 3. Plot the final results
        print("Plotting clusters...")
        plot_clusters(final_clusters_data)

