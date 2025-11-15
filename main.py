from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV
from gaussian_mixture import *
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
        param_grid = {
            "n_components": [2, 3, 4, 5],
            "covariance_type": ["full"], #fixed
            "init_params": ["kmeans", "k-means++", "random", "random_from_data"],
            "n_init": [5],
            "random_state": [2]
        }
        grid_search = GridSearchCV(
            GaussianMixture(), param_grid=param_grid, scoring=gmm_bic_score
        )
        grid_search.fit(df)

        #####for visualizing gmm:
        #grid_search.best_estimator_ holds the best model that we want to plot.
        # apply pca to data because it is high dimensional and we can only plot 2D
        # Use .predict() of our best model on the data shrinked by pca
        #calculate confidence elipses
        #build plot

        output_table, output_plt = gmm_build_output(grid_search) #for parameter comparison

        #export table
        output_table.to_csv(os.path.join(algorithm, dataset, 'final_table.csv'), index=False)

        # export plot
        output_plt.fig.suptitle('Testing different GMM')
        output_plt.savefig(os.path.join(algorithm, dataset, 'final_plot.png'))




if __name__ == "__main__":
    dataset = ['hepatitis', 'hypothyroid', 'heart-statlog'][2]
    algorithm = ['agglomerative', 'gaussian_mixture'][1]

    ### for agglomerative
    n_clusters = [2, 3, 4, 5]
    similarity_metrics = ['euclidean', 'cosine']
    linkage = ['complete', 'average', 'single']

    #config dictionary
    parameters = {'n_clusters': n_clusters,
                  'similarity_metric': similarity_metrics,
                  'linkage': linkage}

    main(dataset, algorithm, parameters)