import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from K_Means import KMeans
from scipy.spatial.distance import cdist



####################################
## Far Efficient K-Means --> FEKM ##
####################################

class FEKM(KMeans): # Inherit Class from KMeans, whe'll only change how we initilize the centroids
    def __init__(self, k, X, metric='euclidean'):
        super().__init__(k, X, metric=metric) # Perform the KMeans class init

    def initialize_centroids(self):
        centers = []
        n_samples = self.X.shape[0]
        available_points = np.ones(n_samples, dtype=bool) # Use data that is not been classified

        # Compute the matrix with all the distances --> points to all points
        all_dists = cdist(self.X, self.X) # Use default distance --> euclidean

        # Get the indices of the maximum indices
        np.fill_diagonal(all_dists, -1) # Ensure we do not get point against itself
        idx1, idx2 = np.unravel_index(np.argmax(all_dists), all_dists.shape) # Get the farthest points indices

        # Initialize the first centers
        center1 = self.X[idx1]
        center2 = self.X[idx2]

        # Cluster for all the points
        cluster_1 = []
        cluster_2 = []

        # Assign and remove points, up to the next threshold
        threshold_limit = int(0.5 * (n_samples / self.k))

        # We iterate through the data up to the threshold
        for i in range(threshold_limit):
            # Calculate distance of the point to both centers
            if self.metric == 'euclidean':
                d1 = np.linalg.norm(self.X[i] - center1)
                d2 = np.linalg.norm(self.X[i] - center2)

            elif self.metric == 'mahnattan':
                d1 = np.sum(np.abs(self.X[i] - center1))
                d2 = np.sum(np.abs(self.X[i] - center2))

            # Assign to closer cluster and remove from the available dataset
            if d1 <= d2:
                cluster_1.append(self.X[i])
            else:
                cluster_2.append(self.X[i])
            
            available_points[i] = False

        # Update centers based on points means
        if len(cluster_1) > 0:
            center1 = np.mean(cluster_1, axis=0)
        if len(cluster_2) > 0:
            center2 = np.mean(cluster_2, axis=0)
            
        centers.append(center1)
        centers.append(center2)

        # Define remaining centers (k-2)
        for _ in range(self.k - 2):
            # Only look at points that haven't been checked removed
            X_remaining = self.X[available_points]
            
            if len(X_remaining) == 0:
                break
                
            centers_array = np.array(centers)
            
            # Distances from remaining points to existing centers
            dists_remaining = cdist(X_remaining, centers_array)
            
            # Find minimum distance to the centers for each point 
            min_dists = np.min(dists_remaining, axis=1)
            
            # Select the maximum of the minimums
            max_dist_idx = np.argmax(min_dists)
            
            # Add to centers
            centers.append(X_remaining[max_dist_idx])


        for idx, c in enumerate(centers):
            self.centroids[idx] = {
                'center': c,
                'points': [],
            }


if __name__ == "__main__":
    # LOAD DATA
    dataset = './data/hypothyroid.preprocessed.csv'
    # dataset = './data/hepatitis.preprocessed.csv'
    # dataset = './data/heart-statlog.preprocessed.csv'

    try:
        data = pd.read_csv(dataset, header=0)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset}")
        exit()

    # PREPROCESSING
    X_features = data.drop(columns=['Class'])
    if 'TBG' in X_features.columns: X_features = X_features.drop(columns=['TBG']) # Conflitive
    X_data = X_features.values.astype(float)

    
    # RUN FEKM
    print("\n--- Running Far Efficient K-Means (FEKM) ---")
    k = 2
    model = FEKM(k=k, X=X_data)
    
    # Fit
    import time
    # initial = time.time()
    labels, centers = model.fit(max_iterations=500, tolerance=1e-10)
    # print(time.time()-initial)

    # PRINT RESULTS
    print("\nCluster Sizes:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for i, label in enumerate(unique_labels):
        print(f"  Cluster {label}: {counts[i]} points")
        
    print(f"\nCentroids finalized at shape: {centers.shape}")

