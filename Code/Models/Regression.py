from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error
from yellowbrick.regressor import PredictionError
import pickle

def hyperParameter(x_train, y_train, score, model, cycles, **params):
    """
    This function fits multiple hyperparameters to find the optimal combinationf.

    Parameters:
    - x_train (df): training features model
    - y_train (df): training target model
    - score (str): scoring metric used to find the best model 
    - model (model): model that needs to be evaluated
    - cycles (int): number of iterations to test the model
    - params: dict with optimal model hyperparameters

    Returns:
    - Dict with the optimal combination of parameters
    - Score of the best model
    """

    #Call on the the hyper parameter fitting modle
    hyp = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=cycles, scoring=score, n_jobs=4, cv=10,
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

def modelConstruction(model_dir, model_name, model, x_train, y_train, x_eval, y_eval, score, train_dates, kf, cycles, visualization, **params):
    """
    This function trains a linear regression model

    Parameters:
    - model_dir (str): directory where model has to be saved
    - model_name (str): name of the model
    - model (model): model that needs to be evaluated
    - x_train (df): training features model
    - y_train (df): training target model
    - x_eval (df): test features model
    - y_eval (df): test target model
    - score (str): sklearn standard scoring metric used by model
    - train_dates (list): dates present in training model 
    - kf (model): used to split training dates into training and test k times
    - cycles (int): number of iterations to test the model
    - visualization (bool): whether you want a scatter model of evaluation results
    - params: dict with optimal model hyperparameters

    Returns: Dict containing all metrics of hyperparameter, training and evaluation of the model
    """

    results_dict = {}

    best_params, best_score = hyperParameter(
        x_train, y_train, score, model, cycles, **params)
    results_dict["Hyper R2 Score"] = best_score

    train_score, train_rmse, model = trainModel(
        x_train, y_train, train_dates, kf, model, **params)
    results_dict["Train R2 Score"] = train_score
    results_dict["Train RMSE Score"] = train_rmse

    eval_score, eval_mse = evalModel(
        model, x_eval, y_eval, visualization, x_train, y_train)
    results_dict["Test R2 Score"] = eval_score
    results_dict["Test RMSE Score"] = eval_mse

    filename = "{0}{1}_model.sav".format(model_dir, model_name)
    pickle.dump(model, open(filename, 'wb'))

    return results_dict
