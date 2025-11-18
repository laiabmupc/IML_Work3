import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix



#############
## K-MEANS ##
#############

class KMeans:
    def __init__(self, k, X, distance='euclidean', max_iterations=500, tolerance=0.0001):
        self.k = k  # Number of clusters
        self.X = X  # Dataset
        self.metric = distance # distance metric
        self.max_iterations = max_iterations # maximum number of iterations
        self.tolerance = tolerance # convergence threshold
        self.centroids = {} # Store the centroids information --> center & sample values contained
        self.cluster_assignments = np.zeros(self.X.shape[0], dtype=int) # To store the cluster assignment for each data point

    def initialize_centroids(self):
        # Randomly select 'k' instances (indices)
        random_indices = np.random.choice(self.X.shape[0], self.k, replace=False)

        # Store the 'k' selected instances as centroids
        for idx, random_idx in enumerate(random_indices):
            center = self.X[random_idx].copy()
            self.centroids[idx] = {'center': center, 'points': np.array([])}

    def compute_distance(self, x, y):
        # x --> sample
        # y --> centroid
        # distance --> euclidean or manhattan
        try:
            if self.metric == 'euclidean':
                return np.sqrt(np.sum(np.power(x - y, 2)))

            elif self.metric == 'manhattan':
                return np.sum(np.abs(x - y))

        except AttributeError:  # Raised if an attribute reference or assignment fails.
            raise Exception('Metric must be Euclidean or Manhattan')


    def compute_centroids(self):
        """
        Recompute centroids based on the new points
        """
        for idx in range(self.k):
            centroid_points = self.centroids[idx]['points']

            if centroid_points.size == 0:  # Reinitialize this centroid to a random data point
                random_idx = np.random.randint(self.X.shape[0])
                self.centroids[idx]['center'] = self.X[random_idx].copy()
            else:
                re_computed_center = centroid_points.mean(axis=0)  # Compute each dimension mean
                self.centroids[idx]['center'] = re_computed_center

    def assign_centroids(self):
        """
        Assign samples to their nearest centroid
        """
        # Delete previous assigned points
        for centroid_idx in self.centroids:
            self.centroids[centroid_idx]['points'] = np.array([])

        for i, sample in enumerate(self.X): # Iterate with an index to track assignments
            # Compute the distance between the sample and all current centroids
            distances = np.array([self.compute_distance(sample, self.centroids[centroid_idx]['center']) for centroid_idx in self.centroids])
            # Retrieve the index of the closest centroid to the sample
            best_centroid_idx = np.argmin(distances)
            # Assign sample to the nearest centroid
            self.centroids[best_centroid_idx]['points'] = np.hstack((self.centroids[best_centroid_idx]['points'], sample))
            # Store the cluster assignment for this data point (by its index)
            self.cluster_assignments[i] = best_centroid_idx

    # Helper function
    def get_centers(self):
        return {idx: self.centroids[idx]['center'] for idx in self.centroids}

    # Fit Function
    def fit(self):
        self.initialize_centroids()  # Initialize centroids
        self.assign_centroids()  # Assign points to initial centroids

        for i in range(self.max_iterations):
            old_centroids_centers = self.get_centers()

            self.compute_centroids()
            self.assign_centroids()

            new_centroids = self.get_centers()

            total_divergence = 0
            for idx in range(len(old_centroids_centers)):
                old_center = old_centroids_centers[idx]  # Get array
                new_center = new_centroids[idx]

                total_divergence += self.compute_distance(old_center, new_center)

            if total_divergence < self.tolerance:
                print(f"Algorithm converged after {i + 1} iterations.")
                # Return both the centroid information and the assignment list
                return self.centroids, self.cluster_assignments

        # Return both the centroid information and the assignment list
        return self.centroids, self.cluster_assignments






if __name__ == "__main__":

    # Choose dataset
    # dataset = './data/heart-statlog.preprocessed.csv'
    dataset = './data/hypothyroid.preprocessed.csv'
    # dataset = './data/hepatitis.preprocessed.csv'

    # Load data from the CSV file
    try:
        data = pd.read_csv(dataset, header=0)
    except FileNotFoundError:
        print("Error: DATASET not found.")
        print("Please make sure the file is in the same directory as the script.")
        exit()

    # Separate the features from the labels
        # Use column names (all columns EXCEPT the last one, 'Class')
    X_features = data.drop(columns=['Class'])
    X_features = X_features.drop(columns=['TBG']) #Null -->  hypothyroid DATASET

    # Use the 'Class' column for labels
    y_labels = data['Class']

    # Convert features to a NumPy array FLOATS (if not won't work)
    X_data = X_features.values.astype(float)

    # Create and fit your K-Means model
    k = 2 # Since we know there are 2 classes
    kmeans_model = KMeans(k=k, X=X_data, max_iterations=500, tolerance=1e-10)

    final_clusters_data, cluster_assignments = kmeans_model.fit()

    #############
    ## CHATGPT ##
    #############
    # Print the cluster size results
    print("\n--- Final Cluster Results ---")
    for idx in final_clusters_data:
        print(f"\nCluster {idx}:")
        print(f"  Number of points in cluster: {len(final_clusters_data[idx]['points'])}")

    # Compare cluster assignments to real labels
    print("\n\n--- Cluster vs. Class Comparison (Confusion Matrix) ---")

    confusion_matrix = pd.crosstab(y_labels, cluster_assignments, rownames=['Actual Class'], colnames=['Cluster ID'])

    print(confusion_matrix)

    # Calculate the "accuracy"
    print("\n\n--- K-Means 'Accuracy' Calculation ---")
    print("Note: K-Means doesn't know cluster-class mapping.")
    print("We check possibilities and take the best mapping.")

    # Convert the matrix to a numpy array for easy math
    cm_array = confusion_matrix.values
    print(f"Confusion matrix shape: {cm_array.shape}")

    # Handle cases where we don't get exactly k clusters
    n_clusters = cm_array.shape[1]
    n_classes = cm_array.shape[0]

    if n_clusters == 1:
        # Only one cluster found - compare to majority class
        total_points = np.sum(cm_array)
        majority_correct = np.max(cm_array)
        best_accuracy = majority_correct / total_points
        print(f"Warning: Algorithm converged to only 1 cluster")
        print(f"Best K-Means 'Accuracy': {best_accuracy * 100:.2f}%")
        
    elif n_clusters == 2:
        # Normal case - two clusters
        # Possibility 1: Cluster 0 = 'absent', Cluster 1 = 'present'
        correct_1 = cm_array[0, 0] + cm_array[1, 1]
        # Possibility 2: Cluster 0 = 'present', Cluster 1 = 'absent'
        correct_2 = cm_array[0, 1] + cm_array[1, 0]

        total_points = np.sum(cm_array)
        best_accuracy = max(correct_1, correct_2) / total_points

        print(f"\nTotal points: {total_points}")
        print(f"Possibility 1 (Cluster 0='absent', Cluster 1='present'): {correct_1} correct")
        print(f"Possibility 2 (Cluster 0='present', Cluster 1='absent'): {correct_2} correct")
        print(f"\nBest K-Means 'Accuracy': {best_accuracy * 100:.2f}%")
        
    else:
        print(f"Unexpected number of clusters: {n_clusters}")