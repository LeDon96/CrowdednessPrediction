from sklearn.model_selection import RandomizedSearchCV
import numpy as np
from sklearn.metrics import mean_squared_error
from yellowbrick.regressor import PredictionError
import pickle
import matplotlib.pyplot as plt


def hyperParameter(x_train, y_train, score, model, model_name, cycles, **params):
    """
    This function fits multiple hyperparameters to find the optimal combinationf.

    Parameters:
    - x_train (df): training features model
    - y_train (df): training target model
    - score (str): scoring metric used to find the best model 
    - model (model): model that needs to be evaluated
    - model_name (str): unique ID model
    - cycles (int): number of iterations to test the model
    - params: dict with optimal model hyperparameters

    Returns:
    - Dict with the optimal combination of parameters
    - Score of the best model
    """

    #Call on the the hyper parameter fitting modle
    hyp = RandomizedSearchCV(estimator=model, param_distributions=params, n_iter=cycles, scoring=score, n_jobs=4, cv=10,
                            refit=score, random_state=42)

    #Run hyper parameter fitting
    ## XGB Regressor model only takes values as input
    if model_name == "xgbr":
        model = hyp.fit(x_train.drop(
            columns={"Date"}).values, y_train["CrowdednessCount"].values)
    else:
        model = hyp.fit(x_train.drop(
            columns={"Date"}), y_train["CrowdednessCount"])

    return model.best_params_, model.best_score_

def trainModel(x_train, y_train, train_dates, kf, model, params, model_name, kf_size):
    """
    This function trains and tests the model k times, by splitting the data k times

    Parameters:
    - x_train (df): training features model
    - y_train (df): training target model
    - train_dates (list): dates present in training model
    - kf (model): used to split training dates into training and test k times
    - model (model): model that needs to be evaluated
    - params: dict with optimal model hyperparameters
    - model_name (str): unique ID model
    - kf_size (int): number of splits in KFolds

    Returns:
    - Mean R2 score of all k splits
    - Mean RMSE score of all k splits
    """

    #variables
    mean_score = 0
    mean_rmse = 0

    #If model is baseline, don't include random state and n_jobs
    if model_name != "lr":
        params["random_state"] = 42
        params["n_jobs"] = 4
        model.set_params(**params)
    else:
        model.set_params(**params)

    #Split the train dates into train and test, n times
    for train_index, test_index in kf.split(train_dates):

        #Select rows dataset based on the dates in train_index

        ## XGB Regressor model only takes values as input
        if model_name == "xgbr":
            x = x_train[x_train["Date"].isin(train_dates[train_index])].drop(
                columns={"Date"}).values
            y = y_train[y_train["Date"].isin(
                train_dates[train_index])]["CrowdednessCount"].values
            
            model.fit(x, y)

            x_test = x_train[x_train["Date"].isin(
                train_dates[test_index])].drop(columns={"Date"}).values
            y_test = y_train[y_train["Date"].isin(
                train_dates[test_index])]["CrowdednessCount"].values

            mean_score += model.score(x_test, y_test)

            y_pred_model = model.predict(x_test)
            mean_rmse += np.sqrt(mean_squared_error(y_pred_model,y_test))
        
        else:
            model.fit(x_train[x_train["Date"].isin(train_dates[train_index])].drop(columns={"Date"}),
                      y_train[y_train["Date"].isin(train_dates[train_index])]["CrowdednessCount"])

            mean_score += model.score(x_train[x_train["Date"].isin(train_dates[test_index])].drop(columns={"Date"}),
                                    y_train[y_train["Date"].isin(train_dates[test_index])]["CrowdednessCount"])

            y_pred_model = model.predict(x_train[x_train["Date"].isin(
                train_dates[test_index])].drop(columns={"Date"}))
            mean_rmse += np.sqrt(mean_squared_error(y_pred_model,
                                                    y_train[y_train["Date"].isin(train_dates[test_index])]["CrowdednessCount"]))

    #Take the mean scores over the n iterations
    mean_score /= kf_size
    mean_rmse /= kf_size

    return mean_score, mean_rmse, model


def evalModel(model, x_eval, y_eval, visualization, plot_dir, model_name, x_train, y_train):
    """
    This function evaluates the trained model on unseen data

    Parameters:
    - model (model): model that needs to be evaluated
    - x_eval (df): test features model
    - y_eval (df): test target model
    - visualization (bool): whether you want a scatter model of evaluation results
    - plot_dir (str): directory where plots have to be saved
    - model_name (str): name of the model
    - x_train (df) (optional): training features model. Only needed for visualization
    - y_train (df): training target model. Only needed for visualization

    Returns:
    - R2 score of the model
    - RMSE score of the model
    """

    #Calculate R2 score and RMSE
    ## XGB Regressor model only takes values as input
    if model_name == "xgbr":
        eval_model_score = model.score(x_eval.values, y_eval.values)
        
        y_pred_eval_model = model.predict(x_eval.values)
        eval_model_mse = mean_squared_error(y_pred_eval_model, y_eval.values)
    else: 
        eval_model_score = model.score(x_eval, y_eval)

        y_pred_eval_model = model.predict(x_eval)
        eval_model_mse = mean_squared_error(y_pred_eval_model, y_eval)

    #Visualize the model results
    if visualization == True:
        visualizer = PredictionError(model)
        # Fit the training data to the visualizer

        ## XGB Regressor model only takes values as input
        if model_name == "xgbr":
            visualizer.fit(x_train.drop(
                columns={"Date"}).values, y_train["CrowdednessCount"].values)
            visualizer.score(x_eval.values, y_eval.values)  # Evaluate the model on the test data
        else: 
            visualizer.fit(x_train.drop(columns={"Date"}), y_train["CrowdednessCount"])
            visualizer.score(x_eval, y_eval)  # Evaluate the model on the test data
        visualizer.poof("{0}{1}.png".format(plot_dir, model_name))
        plt.gcf().clear()
        
    return eval_model_score, np.sqrt(eval_model_mse)

def modelConstruction(model_dir, plot_dir, model_name, model, x_train, y_train, x_eval, y_eval, score, train_dates, kf, cycles, visualization, params, kf_size):
    """
    This function trains a linear regression model

    Parameters:
    - model_dir (str): directory where model has to be saved
    - plot_dir (str): directory where plots have to be saved
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
    - kf_size (int): number of splits in KFolds

    Returns: Dict containing all metrics of hyperparameter, training and evaluation of the model
    """

    #Dict to save all the results in
    results_dict = {}

    #Hyper parameter tuning
    best_params, best_score = hyperParameter(
        x_train, y_train, score, model, model_name, cycles, **params)
    results_dict["Hyper R2 Score"] = best_score
    results_dict["Model Parameters"] = best_params

    #Train the model
    train_score, train_rmse, model = trainModel(
        x_train, y_train, train_dates, kf, model, best_params, model_name, kf_size)
    
    #Save results training
    results_dict["Train R2 Score"] = train_score
    results_dict["Train RMSE Score"] = train_rmse

    #Evaluate the model
    eval_score, eval_mse = evalModel(
        model, x_eval, y_eval, visualization, plot_dir, model_name, x_train, y_train)
    
    #Save results evaluation
    results_dict["Test R2 Score"] = eval_score
    results_dict["Test RMSE Score"] = eval_mse

    #Save the model
    filename = "{0}{1}_model.sav".format(model_dir, model_name)
    pickle.dump(model, open(filename, 'wb'))

    return results_dict
