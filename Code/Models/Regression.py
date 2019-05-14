from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error
from yellowbrick.regressor import PredictionError

def hyperParameter(x_train, y_train, score, model, **params):
    """
    This function fits multiple hyperparameters to find the optimal combinationf.

    Parameters:
    - x_train (df): training features model
    - y_train (df): training target model
    - score (str): scoring metric used to find the best model 
    - model (model): model that needs to be evaluated
    - params: dict with optimal model hyperparameters

    Returns:
    - Dict with the optimal combination of parameters
    - Score of the best model
    """

    #Call on the the hyper parameter fitting modle
    hyp = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=8, scoring=score, n_jobs=4, cv=10,
                            random_state=42, refit=score)

    #Run hyper parameter fitting
    model_model = hyp.fit(x_train.drop(
        columns={"Date"}), y_train["CrowdednessCount"])

    return model_model.best_params_, model_model.best_score_

def trainModel(x_train, y_train, train_dates, kf, model, **params):
    """
    This function trains and tests the model k times, by splitting the data k times

    Parameters:
    - x_train (df): training features model
    - y_train (df): training target model
    - train_dates (list): dates present in training model
    - kf (model): used to split training dates into training and test k times
    - model (model): model that needs to be evaluated
    - params: dict with optimal model hyperparameters

    Returns:
    - Mean R2 score of all k splits
    - Mean RMSE score of all k splits
    """

    mean_score = 0
    mean_rmse = 0

    for train_index, test_index in kf.split(train_dates):
        model.fit(x_train[x_train["Date"].isin(train_dates[train_index])].drop(columns={"Date"}),
                y_train[y_train["Date"].isin(train_dates[train_index])]["CrowdednessCount"])

        mean_score += model.score(x_train[x_train["Date"].isin(train_dates[test_index])].drop(columns={"Date"}),
                                y_train[y_train["Date"].isin(train_dates[test_index])]["CrowdednessCount"])

        y_pred_model = model.predict(x_train[x_train["Date"].isin(
            train_dates[test_index])].drop(columns={"Date"}))
        mean_rmse += np.sqrt(mean_squared_error(y_pred_model,
                                                y_train[y_train["Date"].isin(train_dates[test_index])]["CrowdednessCount"]))

    mean_score /= 10
    mean_rmse /= 10

    return mean_score, mean_rmse, model


def evalModel(model, x_eval, y_eval, visualization, x_train, y_train):
    """
    This function evaluates the trained model on unseen data

    Parameters:
    - model (model): model that needs to be evaluated
    - x_eval (df): test features model
    - y_eval (df): test target model
    - visualization (bool): whether you want a scatter model of evaluation results
    - x_train (df) (optional): training features model. Only needed for visualization
    - y_train (df): training target model. Only needed for visualization

    Returns:
    - R2 score of the model
    - RMSE score of the model
    """

    eval_model_score = model.score(x_eval, y_eval)
    
    y_pred_eval_model = model.predict(x_eval)
    eval_model_mse = mean_squared_error(y_pred_eval_model, y_eval)

    if visualization == True:
        visualizer = PredictionError(model)
        # Fit the training data to the visualizer
        visualizer.fit(x_train.drop(columns={"Date"}), y_train["CrowdednessCount"])
        visualizer.score(x_eval, y_eval)  # Evaluate the model on the test data
        g = visualizer.poof()

    return eval_model_score, np.sqrt(eval_model_mse)
