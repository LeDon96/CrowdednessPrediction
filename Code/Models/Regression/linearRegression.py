from sklearn.linear_model import LinearRegression
from sklearn.model_selection import RandomizedSearchCV

def lrHyperParameter(x_train, y_train):

    #Call on model
    base = LinearRegression()

    #Parameters to test
    fit_intercept = [True, False]
    normalize = [True, False]
    copy_X = [True, False]

    #Save parameters to dict
    params = {"fit_intercept": fit_intercept,
            "normalize": normalize,
            "copy_X": copy_X}

    #Errors to train on
    scores = ["r2"]

    #Call on the the hyper parameter fitting modle
    hyp = RandomizedSearchCV(estimator=base, param_distributions=params, n_iter=8, scoring=scores, n_jobs=2, cv=10,
                            random_state=42, refit="r2")

    #Run hyper parameter fitting
    base_model = hyp.fit(x_train.drop(
        columns={"Date"}), y_train["CrowdednessCount"])

    return base_model.best_params_, base_model.best_score_
