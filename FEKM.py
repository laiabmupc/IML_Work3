import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from K_Means import KMeans 



####################################
## Far Efficient K-Means --> FEKM ##
####################################

class FEKM(KMeans): # Inherit Class from KMeans, whe'll only change how we initilize the centroids
    def __init__(self, k, X, metric='euclidean'):
        super().__init__(k, X, metric=metric) # Perform the KMeans class init

    def initialize_centroids(self):
        centers = []
        # Compute the center of our dataset (not a real point)
        dataset_mean = self.X.mean(axis=0)

        # Select the point that is farther to the center        
        if self.metric == 'euclidean':
            distances_to_mean = np.linalg.norm(self.X - dataset_mean, axis=1)
        elif self.metric == 'manhattan':
            distances_to_mean = np.sum(np.abs(self.X - dataset_mean), axis=1)
        
        first_idx = np.argmax(distances_to_mean) # select the idx of the closest point to the mean
        centers.append(self.X[first_idx])

        # Compute the  
        for _ in range(self.k - 1):
            # Calculate distance from every point to every current center
            current_centers_array = np.array(centers)
            
            # Distance matrix calculation
            if self.metric == 'euclidean':
                # Efficient calculation of distances to all centers
                dists = np.array([np.linalg.norm(self.X - c, axis=1) for c in centers]).T
            else:
                dists = np.array([np.sum(np.abs(self.X - c), axis=1) for c in centers]).T
            
            # Find the minimum distance to ANY existing center for each point
            min_dists = np.min(dists, axis=1)
            
            # Select the point with the MAXIMUM of these minimum distances (Farthest from the group)
            next_idx = np.argmax(min_dists)
            centers.append(self.X[next_idx])


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
    labels, centers = model.fit(max_iterations=500, tolerance=1e-10)

    # PRINT RESULTS
    print("\nCluster Sizes:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for i, label in enumerate(unique_labels):
        print(f"  Cluster {label}: {counts[i]} points")
        
    print(f"\nCentroids finalized at shape: {centers.shape}")

