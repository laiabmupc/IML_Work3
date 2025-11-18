from gaussian_mixture import gaussian_mixture
from preprocessing import preprocess_data, load_data
from agglomerative_clustering import agglomerative_clustering
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import os
#from gaussian_mixture_antonio import *


def main(dataset, algorithm, parameters):
    ## PREPROCESSING AND DATA LOADING
    preprocessed = os.path.join('.', 'data', dataset + '.preprocessed.csv')
    preprocessed_class = os.path.join('.', 'data', dataset + '_class' + '.preprocessed.csv')
    if not os.path.isfile(preprocessed) or not os.path.isfile(preprocessed_class):
        preprocess_data(dataset)
    df, class_df = load_data(dataset)

    os.makedirs(algorithm, exist_ok=True)
    os.makedirs(os.path.join(algorithm, dataset), exist_ok=True)

    ## CONFIGURATIONS
    configurations = list(ParameterGrid(parameters))
    ## AGGLOMERATIVE CLUSTERING
    if algorithm == 'agglomerative':
        for config in configurations:
            clustering, dendrogram = agglomerative_clustering(df, config)
            out_name = f"{config['linkage']}_{config['n_clusters']}_{config['similarity_metric']}"
            plot_name = f"LINKAGE: {config['linkage']} | K: {config['n_clusters']} | METRIC: {config['similarity_metric']}"
            plt.title(plot_name)
            plt.savefig(os.path.join(algorithm, dataset, out_name + '.png'))
            plt.close()

    ## GAUSSIAN MIXTURE
    if algorithm == 'gaussian_mixture':        
        for config in configurations:
            clustering = gaussian_mixture(df, config)






if __name__ == '__main__':
    dataset = ['hepatitis', 'hypothyroid', 'heart-statlog'][2]
    algorithm = ['agglomerative', 'gaussian_mixture', 'k-means'][1]

    # Build a dictionary of parameter values
    parameters = {}
    if algorithm == 'agglomerative':
        parameters = {'n_clusters': [2, 3, 4, 5],
                      'similarity_metric': ['euclidean', 'cosine'],
                      'linkage': ['complete', 'average', 'single']}

    elif algorithm == 'gaussian_mixture':
        parameters = {'n_components': [2, 3, 4, 5],
                      'covariance_type': ['full'], # fixed
                      'init_params': ['kmeans', 'k-means++', 'random', 'random_from_data'],
                      'n_init': [5],
                      'random_state': [1, 2, 3, 4]}

    elif algorithm == 'k-means':
        parameters = {'k': [2, 3, 4, 5],
                      'max_iter': [100]}
    
    main(dataset, algorithm, parameters)