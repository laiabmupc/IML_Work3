import pandas as pd
import seaborn as sns

def gmm_build_output(grid_search):
    """
    https://scikit-learn.org/stable/auto_examples/mixture/plot_gmm_selection.html#sphx-glr-auto-examples-mixture-plot-gmm-selection-py
    build table to display number of components and their BIC score.
    BIC score helps us determine which parameter configuration is best (less is better).
    Additionally, build plot summarizing the experiment.
    """
    df = pd.DataFrame(grid_search.cv_results_)[
        ["param_n_components", "param_init_params", "mean_test_score"]
    ]
    df["mean_test_score"] = -df["mean_test_score"]
    df = df.rename(
        columns={
            "param_n_components": "Number of components",
            "param_init_params": "Initiation method",
            "mean_test_score": "BIC score",
        }
    )
    sorted_df = df.sort_values(by="BIC score")

    #plot
    plt = sns.catplot(
        data=sorted_df,
        kind="bar",
        x="Number of components",
        y="BIC score",
        hue="Initiation method"
    )
    plt.set(yscale="log") #log scale for proper visualization
    return sorted_df, plt

def gmm_bic_score(estimator, X):
    """
    Callable to pass to GridSearchCV that will use the BIC score.
    Function used in the testing of different gaussian mixture params configurations
    """
    # Make it negative since GridSearchCV expects a score to maximize
    return -estimator.bic(X)
