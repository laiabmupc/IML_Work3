from sklearn.mixture import GaussianMixture

def gaussian_mixture(df, config):
    '''
    Performs Expectation-Maximization clustering on the provided dataset.
    Returns the predicted model labels.
    '''
    model = GaussianMixture(n_components=config['n_components'],
                            covariance_type=config['covariance_type'],
                            init_params=config['init_params'],
                            n_init=config['n_init'],
                            random_state=config['random_state'])
    labels = model.fit_predict(df)
    return labels
