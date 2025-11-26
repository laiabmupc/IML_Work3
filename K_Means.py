import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix



#############
## K-MEANS ##
#############

class KMeans:
    def __init__(self, k, X, metric='euclidean'):
        self.k = k  # Number of centroids
        self.X = X  # Dataset
        self.centroids = {} # Store the centroids information --> center & sample values contained
        self.cluster_assignments = np.zeros(self.X.shape[0], dtype=int) # To store the cluster assignment for each data point
        self.inertia = 0 # Sum of squared distances to closest centroid
        self.metric = metric

    def initialize_centroids(self):
        # Random initialization
        random_indices = np.random.choice(self.X.shape[0], self.k, replace=False) # Do not replace to avoid picking the same point twice.

        for idx, random_idx in enumerate(random_indices):
            center = self.X[random_idx].copy()
            points = []
            centroid = {
                'center': center,
                'points': points,
            }
            self.centroids[idx] = centroid

    def compute_distance(self, x, y, metric='euclidean'):
        # x --> sample
        # y --> centroid
        # distance --> euclidean or manhattan
        try:
            if metric == 'euclidean':
                return np.sqrt(np.sum((x - y) ** 2))

            elif metric == 'manhattan':
                return np.sum(np.abs(x - y))

        except AttributeError:  # Raised if an attribute reference or assignment fails.
            raise Exception('Metric must be Euclidean or Manhattan')


    # Recompute centroids based on the new points
    def compute_centroids(self):
        for idx in range(self.k):
            centroid_points = np.array(self.centroids[idx]['points'])

            if centroid_points.size == 0:  # Reinitialize this centroid to a random data point
                random_idx = np.random.randint(self.X.shape[0])
                self.centroids[idx]['center'] = self.X[random_idx].copy()
                continue
            else:
             self.centroids[idx]['center'] = centroid_points.mean(axis=0)


    # This function assign samples to its nearest centroid
    def assign_centroids(self):

        # Delete previous assigned points
        for centroid_idx in self.centroids:
            self.centroids[centroid_idx]['points'] = []
        
        total_distance_sum = 0

        for i, sample in enumerate(self.X): # Iterate with an index to track assignments
            best_distance = np.inf
            best_centroid_idx = -1
            for centroid_idx in self.centroids:
                # Compute distance
                distance = self.compute_distance(sample, self.centroids[centroid_idx]['center'])

                # Check if the distance is better
                if distance < best_distance:
                    best_distance = distance
                    best_centroid_idx = centroid_idx

            # Assign sample to the nearest centroid
            self.centroids[best_centroid_idx]['points'].append(sample)
            # Store the cluster assignment for this data point (by its index)
            self.cluster_assignments[i] = best_centroid_idx
            # For inertia calculation
            total_distance_sum += best_distance**2
        
        self.inertia = total_distance_sum


    # Helper function
    def get_centers(self):
        return {idx: self.centroids[idx]['center'] for idx in self.centroids}

    # Fit Function
    def fit(self, max_iterations=1000, tolerance=0):
        self.initialize_centroids()  # Initialize centroids
        self.assign_centroids()  # Assign points to initial centroids

        for i in range(max_iterations):
            old_centroids_centers = self.get_centers()

            self.compute_centroids()
            self.assign_centroids()

            new_centroids = self.get_centers()
            
            # Check convergence
            total_divergence = 0
            for idx in range(len(old_centroids_centers)):
                old_center = old_centroids_centers[idx]  # Get array
                new_center = new_centroids[idx]

                total_divergence += self.compute_distance(old_center, new_center, metric='euclidean')

            if total_divergence < tolerance:
                print(f"Algorithm converged after {i + 1} iterations.")
                break

        # Get coordinates
        centroids_coords = np.array([self.centroids[i]['center'] for i in range(self.k)])
        
        # Return LABELS and COORDINATES
        return self.cluster_assignments, centroids_coords


if __name__ == "__main__":
    # LOAD DATA
    dataset = './data/hypothyroid.preprocessed.csv'
    # dataset = './data/hepatitis.preprocessed.csv'
    # dataset = './data/heart-statlog.preprocessed.csv'

    try:
        data = pd.read_csv(dataset, header=0)
    except FileNotFoundError:
        print(f"Error: Dataset not found at {dataset}")
        exit()

    # PREPROCESSING
    X_features = data.drop(columns=['Class'])
    if 'TBG' in X_features.columns: X_features = X_features.drop(columns=['TBG']) # Conflitive
    X_data = X_features.values.astype(float)

    
    # RUN STANDARD K-MEANS
    print("\n--- Running Standard K-Means ---")
    k = 2
    model = KMeans(k=k, X=X_data)
    
    # Fit
    labels, centers = model.fit(max_iterations=500, tolerance=1e-10)
    
    # PRINT RESULTS
    print("\nCluster Sizes:")
    unique_labels, counts = np.unique(labels, return_counts=True)
    for i, label in enumerate(unique_labels):
        print(f"  Cluster {label}: {counts[i]} points")
        
    print(f"\nCentroids finalized at shape: {centers.shape}")