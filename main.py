from preprocessing import preprocess_data, load_data
from agglomerative_clustering import agglomerative_clustering
from sklearn.model_selection import ParameterGrid
import matplotlib.pyplot as plt
import os

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
    for config in configurations:
        ## AGGLOMERATIVE CLUSTERING
        if algorithm == 'agglomerative':
            clustering, dendrogram = agglomerative_clustering(df, config)
            out_name = f"{config['linkage']}_{config['n_clusters']}_{config['similarity_metric']}"
            plot_name = f"LINKAGE: {config['linkage']} | K: {config['n_clusters']} | METRIC: {config['similarity_metric']}"
            plt.title(plot_name)
            plt.savefig(os.path.join(algorithm, dataset, out_name + '.png'))
            plt.close()

        ## GAUSSIAN MIXTURE




if __name__ == "__main__":
    dataset = ['hepatitis', 'hypothyroid', 'heart-statlog'][2]
    algorithm = ['agglomerative', 'gaussian mixture'][0]

    n_clusters = [2, 3, 4, 5]
    similarity_metrics = ['euclidean', 'cosine']
    linkage = ['complete', 'average', 'single']
    parameters = {'n_clusters': n_clusters,
                  'similarity_metric': similarity_metrics,
                  'linkage': linkage}

    main(dataset, algorithm, parameters)