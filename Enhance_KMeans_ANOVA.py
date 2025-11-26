import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from K_Means import KMeans 
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.decomposition import PCA # Needed to visualize high-dimensional data


class KMeans_ANOVA(KMeans):
    def __init__(self, k, X, metric='euclidean', alpha = 0.5):
        super().__init__(k, X, metric=metric) # Perform the KMeans class init
        self.alpha = alpha # Specific hyper-parameter

        # Simulate the Energy (in the real world this would come from the sensors)
        np.random.seed(23)
        self.energies = np.random.uniform(0.1, 1.0, size=self.X.shape[0])
    

    # We now also store the indices, we will later use them to know the Energy
    def assign_centroids(self):
        for centroid_idx in self.centroids:
            self.centroids[centroid_idx]['points'] = []
            self.centroids[centroid_idx]['indices'] = [] # Indices

        for i, sample in enumerate(self.X): # Iterate with an index to track assignments
            best_distance = np.inf
            best_centroid_idx = -1
            for centroid_idx in self.centroids:
                # Compute distance
                distance = self.compute_distance(sample, self.centroids[centroid_idx]['center'], metric=self.metric)
                

                # Check if the distance is better
                if distance < best_distance:
                    best_distance = distance
                    best_centroid_idx = centroid_idx

            # Assign sample to the nearest centroid
            self.centroids[best_centroid_idx]['points'].append(sample)
            self.centroids[best_centroid_idx]['indices'].append(i)
            # Store the cluster assignment for this data point (by its index)
            self.cluster_assignments[i] = best_centroid_idx

     # Recompute centroids based on the GEOMETRIC DISTANCE & HAVING HIGH ENERGY
    def compute_centroids(self):
        for idx in range(self.k):
            indices = self.centroids[idx]['indices']
            points = np.array(self.centroids[idx]['points'])

            if points.size == 0:  # Reinitialize this centroid to a random data point
                random_idx = np.random.randint(self.X.shape[0])
                self.centroids[idx]['center'] = self.X[random_idx].copy()
                continue

            # Compute geometric center --> theoretical best 
            geometric_center = points.mean(axis=0)  # Compute each dimension mean

            # Find the point that is best to be the cluster center
            best_candidate_idx = -1
            max_fitness = -np.inf

            # Calculate max distance for normalization
            if self.metric == 'euclidean':
                dists = np.linalg.norm(points - geometric_center, axis=1)
            elif self.metric == 'manhattan':
                dists = np.sum(np.abs(points - geometric_center), axis=1)

            max_dist = np.max(dists) if np.max(dists) > 0 else 1.0

            # Loop through every member of the cluster
            for i, real_idx in enumerate(indices):
                # Get Energy
                energy = self.energies[real_idx]
                # Get Distance to the geometric center
                dist = dists[i]

                # FITNESS FUNCTION (Based on Harb et al.) --> trade off between distance & energy
                # Normalize distance (1 = center, 0 = edge)
                dist_score = (max_dist - dist) / max_dist
                
                # Weighted score 
                    # HIGH ALPHA --> prioritze ENERGY
                fitness = (self.alpha * energy) + ((1 - self.alpha) * dist_score)

                if fitness > max_fitness:
                    max_fitness = fitness
                    best_candidate_idx = real_idx

            # Update the centroid
            self.centroids[idx]['center'] = self.X[best_candidate_idx].copy()

    # Check if the cluster data is similar enough to aggregate.
    def anova_aggregation(self, threshold_factor=0.5):
        # Checks intra-cluster variance to decide on aggregation.
        results = {}
        
        # Calculate Global Variance for context
        global_var = np.mean(np.var(self.X, axis=0))
        # Compute threshold based on the variance
        threshold = global_var * threshold_factor

        for idx in self.centroids:
            points = np.array(self.centroids[idx]['points'])
            
            if len(points) < 2:
                results[idx] = "RAW (Not enough data)"
                continue

            # Mean of variances across dimensions
            variance = np.mean(np.var(points, axis=0))
            
            if variance < threshold:
                decision = "AGGREGATE (Data is similar)"
            else:
                decision = "SEND RAW (High Variance)"

            print(f"Cluster {idx}: Var={variance:.4f} -> {decision}")
            results[idx] = decision
            
        return results


from sklearn.preprocessing import StandardScaler # <--- 1. Import this

if __name__ == "__main__":
    # LOAD DATA
    dataset = './data/hypothyroid.preprocessed.csv'

    try:
        data = pd.read_csv(dataset, header=0)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset}")
        exit()

    # PREPROCESSING
    X_features = data.drop(columns=['Class'])
    if 'TBG' in X_features.columns: X_features = X_features.drop(columns=['TBG'])
    X_data = X_features.values.astype(float)

    # --- CRITICAL FIX: SCALE THE DATA ---
    # This ensures variance is comparable across all features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data) 
    # ------------------------------------
    
    # RUN ENHANCED K-MEANS
    print("\n--- Running Enhanced K-Means (Harb et al.) ---")
    k = 2
    
    # Use X_scaled instead of X_data
    model = KMeans_ANOVA(k=k, X=X_scaled, alpha=0.5) 
    
    # Fit
    labels, centers = model.fit(max_iterations=500, tolerance=1e-10)

    # PRINT RESULTS
    print("\nCluster Sizes:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for i, label in enumerate(unique_labels):
        print(f"  Cluster {label}: {counts[i]} points")
        
    print(f"\nCentroids finalized at shape: {centers.shape}")

    # ANOVA AGGREGATION CHECK
    print("\n--- Running ANOVA Check ---")    

    # Now the threshold is meaningful because all features have variance ~1.0
    # A factor of 0.5 means: "Aggregate if cluster is half as wide as the whole dataset"
    model.anova_aggregation(threshold_factor=0.5)

