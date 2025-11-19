import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix



####################################
## Far Efficient K-Means --> FEKM ##
####################################

class FEKM:
    def __init__(self, k, X):
        self.k = k  # Number of centroids
        self.X = X  # Dataset
        self.centroids = {} # Store the centroids information --> center & sample values contained
        self.cluster_assignments = np.zeros(self.X.shape[0], dtype=int) # To store the cluster assignment for each data point

    def initialize_centroids(self):
        centers = []
        # Compute the center of our dataset (not a real point)
        dataset_mean = self.X.mean(axis=0)

        # Select the point that is closer to the center
        distances_to_mean = [np.sqrt(np.sum(np.power(sample - dataset_mean, 2))) for sample in self.X]
        first_idx = np.argmax(distances_to_mean) # select the idx of the closest point to the mean
        centers.append(self.X[first_idx])

        # Compute the  
        for _ in range(self.k - 1):
            min_distances = []
            
            for sample in self.X:
                # Calculate distances from sample to all existing centers
                dists_to_current_centers = [np.sqrt(np.sum(np.power(sample - c, 2))) for c in centers]
                
                # Find the distance to the closest center
                min_dist_to_any_center = min(dists_to_current_centers)
                min_distances.append(min_dist_to_any_center)
            
            # Select the point that has the maximum of the minimum distances --> FARTHER
            next_idx = np.argmax(min_distances)
            centers.append(self.X[next_idx])


        for idx, c in enumerate(centers):
            self.centroids[idx] = {
                'center': c,
                'points': [],
            }


    def compute_distance(self, x, y, metric='euclidean'):
        # x --> sample
        # y --> centroid
        # distance --> euclidean or manhattan
        try:
            if metric == 'euclidean':
                return np.sqrt(np.sum(np.power(x - y, 2)))

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

            re_computed_center = centroid_points.mean(axis=0)  # Compute each dimension mean
            self.centroids[idx]['center'] = re_computed_center


    # This function assign samples to its nearest centroid
    def assign_centroids(self):

        # Delete previous assigned points
        for centroid_idx in self.centroids:
            self.centroids[centroid_idx]['points'] = []

        for i, sample in enumerate(self.X): # Iterate with an index to track assignments
            best_distance = np.inf
            best_centroid_idx = -1
            for centroid_idx in self.centroids:
                # Compute distance
                distance = self.compute_distance(sample, self.centroids[centroid_idx]['center'], metric='euclidean')

                # Check if the distance is better
                if distance < best_distance:
                    best_distance = distance
                    best_centroid_idx = centroid_idx

            # Assign sample to the nearest centroid
            self.centroids[best_centroid_idx]['points'].append(sample)
            # Store the cluster assignment for this data point (by its index)
            self.cluster_assignments[i] = best_centroid_idx


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

            total_divergence = 0
            for idx in range(len(old_centroids_centers)):
                old_center = old_centroids_centers[idx]  # Get array
                new_center = new_centroids[idx]

                total_divergence += self.compute_distance(old_center, new_center, metric='euclidean')

            if total_divergence < tolerance:
                print(f"Algorithm converged after {i + 1} iterations.")
                break

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
    kmeans_model = FEKM(k=k, X=X_data)

    final_clusters_data, cluster_assignments = kmeans_model.fit(max_iterations=500, tolerance=1e-10)

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