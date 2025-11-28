import numpy as np
import pandas as pd
from scipy.stats import bartlett
from sklearn.preprocessing import StandardScaler

####################################
## ENHANCED K-MEANS (ANOVA-BASED) ##
####################################
class ANOVA_KMeans:
    def __init__(self, X, alpha=0.05):
        self.X = X
        self.alpha = alpha # Significance level Bartlett test.
        self.k_final = 0
        self.labels = np.zeros(self.X.shape[0], dtype=int) # # Initialize labels array with zeros --> to which cluster each point belongs to
        self.centroids = {}

    # Bartlett Test --> determine if the points in the same cluster are redundant (similar) enough. 
    def variance_study(self, indices):
        # Only 1 point case
        if len(indices) == 1: 
            return True
        
        # Extrcat data belonging to this specific cluster
        cluster_data = self.X[indices]
        
        try:
            # Bartlett --> null hypothesis = all input samples have equal variances.
            stat, p_value = bartlett(*cluster_data) # Unpack rows as separate arguments.
        # Handle cases where variance is 0
        except ValueError:
            return True
        
        return p_value > self.alpha


    # If a cluster has high variance, split it into 2 sub-clusters
    def internal_kmeans_split(self, indices):
        subset = self.X[indices]
        np.random.seed(23)
        
        # Randomly initialize 2 centroids from the current subset
        center_idxs = np.random.choice(subset.shape[0], 2, replace=False)        
        centers = subset[center_idxs]
        subset_labels = np.zeros(subset.shape[0], dtype=int)
        
        # Run a FAST K-Means --> fixed 10 iterations for speed
        for _ in range(10):
            # Compute Euclidean Distance between points and 2 centroids
            dists = np.linalg.norm(subset[:, None] - centers, axis=2)
            new_labels = np.argmin(dists, axis=1)
            
            # Check convergence
            if np.array_equal(subset_labels, new_labels): 
                break
            subset_labels = new_labels
            
            # Update centroids --> mean of assigned points
            for i in range(2):
                if np.any(subset_labels == i):
                    centers[i] = subset[subset_labels == i].mean(axis=0)
        
        # Return the indices separated into two groups
        return [indices[subset_labels == 0], indices[subset_labels == 1]]


    # Recursive Divisive Clustering
    def fit(self, max_iterations=None, tolerance=None):
        # Max iterations is not used as default -> we are based on variance.
        all_indices = np.arange(self.X.shape[0])
        Q = [all_indices] # Queue for Breadth-First Search splitting
        final_clusters_indices = []
        
        # Loop until queue is empty
        while Q:
            current_indices = Q.pop(0) # get next cluster to analyze
            if len(current_indices) == 0: 
                continue

            # Check if this cluster is redundant
            is_redundant = self.variance_study(current_indices)
            
            # If variance is low the cluster is saved
            if is_redundant:
                final_clusters_indices.append(current_indices)
            # If variance is high, split it into 2 sub-clusters
            else:
                sub_clusters = self.internal_kmeans_split(current_indices)
                Q.extend(sub_clusters) # Add new sub-clusters back to the Queue to be tested again

        # Process Final Results
        self.k_final = len(final_clusters_indices)
        self.final_centroids = np.zeros((self.k_final, self.X.shape[1])) 
        
        # Assign final labels and compute final centroids
        for label_id, indices in enumerate(final_clusters_indices):
            # Update labels map
            self.labels[indices] = label_id
            
            # Compute centroid --> mean
            center_coords = self.X[indices].mean(axis=0)
            self.final_centroids[label_id] = center_coords
            
            # Store in dictionary to be returnfed
            self.centroids[label_id] = {
                'center': center_coords,
                'indices': indices,
                'size': len(indices)
            }

        # Comptue Data Reduction 
        reduction = (self.X.shape[0] - self.k_final) / self.X.shape[0] * 100
        print(f'Algorithm converged. Found {self.k_final} clusters.')
        print(f'Data reduction {round(reduction, 2)}%')
        
        # Return to which centroid each sample belongs and centroid coordinates
        return self.labels, self.final_centroids



if __name__ == "__main__":
    # LOAD DATA
    dataset = './data/heart-statlog.preprocessed.csv'
    # dataset = './data/hepatitis.preprocessed.csv'
    # dataset = './data/hypothyroid.preprocessed.csv'

    try:
        data = pd.read_csv(dataset, header=0)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset}")
        exit()

    # PREPROCESSING
    X_features = data.drop(columns=['Class'])
    if 'TBG' in X_features.columns: X_features = X_features.drop(columns=['TBG'])
    X_data = X_features.values.astype(float)

    # SCALING
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_data) 
    
    # RUN
    print("\n Running Enhanced K-Means (Harb et al.)")
    model = ANOVA_KMeans(X=X_scaled, alpha=0.05)
    
    # Fit
    labels, centers = model.fit()
    
    # PRINT RESULTS WITH VARIANCE
    unique_labels, counts = np.unique(labels, return_counts=True)
    
    # Sort indices by cluster size (descending)
    sorted_indices = np.argsort(-counts)
    
    for i in sorted_indices[:20]: # Show only top 20
        label = unique_labels[i]
        count = counts[i]
        
        # 1. Get all points belonging to this cluster
        cluster_points = X_scaled[labels == label]
        
        # 2. Calculate Variance for this cluster
        # We take the variance of each feature, then average them to get a single score
        cluster_var = np.mean(np.var(cluster_points, axis=0))
        
        print(f"  Cluster {label}: {count} points | Avg Variance: {cluster_var:.4f}")
    
    print(labels)
    print(centers)