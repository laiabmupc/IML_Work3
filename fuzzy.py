import numpy as np

class s_FCM():
    def __init__(self, X, c, m=2, alpha=0.2, max_iter=100, epsilon=0.01, seed=1):
        # Set a seed for reproducibility
        np.random.seed(seed)
        # Input data
        self.X = X.to_numpy(dtype=float)
        # Number of clusters
        self.c = c
        # Fuzzy exponent
        if m < 1:
            raise ValueError("m must be > 1")
        self.m = m
        # Maximum number of iterations
        self.max_iter = max_iter
        # Define the epsilon (tolerance) threshold
        self.epsilon = epsilon
        # Define alpha for the suppression
        self.alpha = alpha

        # Initialize the centroids randomly
        self.v = self.initialize_centroids()
        # Initialize the universe matrix
        self.U = self.membership_matrix()
    

    def initialize_centroids(self):
        '''
        Randomly initialize c centroids
        '''
        inds = np.random.choice(self.X.shape[0], size=self.c, replace=False)
        return self.X[inds].astype(float)


    def distance_sq(self):
        '''
        Computes the distance between the samples X and the centroids v
        Returns the distance squared
        '''
        x_reshaped = self.X.reshape(self.X.shape[0], 1, self.X.shape[1]) # (n_samples, 1, n_features)
        v_reshaped = self.v.reshape(1, self.v.shape[0], self.v.shape[1]) # (1, n_centroids, n_features)
        return np.sum((x_reshaped - v_reshaped)**2, axis=2)


    def membership_matrix(self):
        if self.m == 1:
            exp = 1 # VERIFICAR EL VALOR QUE HA DE TENIR L'EXPONENT QUAN m = 1
        else:
            exp = -2/(self.m-1) # add 1e-10 to avoid dividing by 0
        d_sq = self.distance_sq()
        d_sq = np.maximum(d_sq, 1e-10) # add 1e-10 to avoid numerical conflicts in the numerator
        numerator = d_sq**exp
        denominator = np.sum(numerator, axis=1, keepdims=True)
        # (n_samples, n_centroids)
        U_transposed = numerator/denominator
        # (n_centroids, n_samples)
        self.U = U_transposed.T
        # Apply the suppression step (s-FCM)
        self.suppression()


    def suppression(self):
        '''
        Apply the s-FCM suppression to the membership matrix
            winner: 1 - alpha + alpha * Uik
            others: alpha*Uik
        '''
        for col in range(self.U.shape[1]):
            # Get the whole column
            sample_col = self.U[:, col]
            winner_index = np.argmax(sample_col)
            suppressed_col = self.alpha*sample_col
            suppressed_col[winner_index] = 1-self.alpha + self.alpha*sample_col[winner_index]
            self.U[:, col] = suppressed_col


    def centroids_computation(self):
        '''
        Update the centroids according to the membership matrix (with applied suppression)
        '''
        U_m = self.U ** self.m 
        numerator = U_m @ self.X
        denominator = np.sum(U_m, axis=1, keepdims=True)
        # Update the centroids
        self.v = numerator / denominator


    def fit(self):
        '''
        Apply s-FCM algorithm
        '''
        n_iter = 0
        while n_iter < self.max_iter:
            v_prev = self.v.copy()
            # STEP 1: update the membership matrix
            self.membership_matrix() # s-FCM suppression alreaady performed in this call
            # STEP 2: compute the corresponding centroids
            self.centroids_computation()
            if np.linalg.norm(self.v - v_prev) < self.epsilon:
                return self.U, self.v
            n_iter += 1
        return self.U, self.v