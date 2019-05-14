import Regression as reg
from TrainTestSplit import trainTestSplit
import pandas as pd
from sklearn.model_selection import KFold
import pickle

#Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def regressionModel(model_dir, model_name, model, x_train, y_train, x_eval, y_eval, score, train_dates, kf, visualization, **params):
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
    - visualization (bool): whether you want a scatter model of evaluation results
    - params: dict with optimal model hyperparameters

    Returns: Dict containing all metrics of hyperparameter, training and evaluation of the model
    """

    results_dict = {}

    best_params, best_score = reg.hyperParameter(x_train, y_train, score, model, **params)
    results_dict["Hyper R2 Score"] = best_score

    train_score, train_rmse, model = reg.trainModel(x_train, y_train, train_dates, kf, model, **params)
    results_dict["Train R2 Score"] = train_score
    results_dict["Train RMSE Score"] = train_rmse

    eval_score, eval_mse = reg.evalModel(model, x_eval, y_eval, visualization, x_train, y_train)
    results_dict["Test R2 Score"] = eval_score
    results_dict["Test RMSE Score"] = eval_mse

    filename = "{0}{1}_model.sav".format(model_dir, model_name)
    pickle.dump(model, open(filename, 'wb'))

    return results_dict

def regressionConstruction(full_df, size, stations, model_dir):

    score = "r2"
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    metrics_dict = {}
    visualization = True

    x_train, y_train, x_eval, y_eval, train_dates = trainTestSplit(
        full_df, size, stations)

    #Linear Regression
    # lr = LinearRegression()
    # params = {"fit_intercept": [True, False],
    #           "normalize": [True, False],
    #           "copy_X": [True, False]}

    # model_name = "lr"

    # metrics_dict["Linear Regression"] = regressionModel(model_dir, model_name,
    #     lr, x_train, y_train, x_eval, y_eval, score, train_dates, kf, visualization, **params)
    
    #Random Forrest Regressor
    rfg = RandomForestRegressor()

    params = {"n_estimators": list(range(100, 400, 25)),
              "criterion": ["mse"],
              "max_features": ["log2", "auto", None],
              "bootstrap": [True],
              "oob_score": [True, False]}

    model_name = "rfg"

    metrics_dict["Random Forest Regressor"] = regressionModel(model_dir, model_name,
                                                        rfg, x_train, y_train, x_eval, y_eval, score, train_dates, kf, visualization, **params)

    #XGBoost Regressor
    xgbr = xgb.XGBRegressor()

    params = {"learning_rate": list(range(0, 0,30, 0.05)),
              "n_estimators": list(range(100, 400,25)),
              "booster": ["gbtree"],
              "objective": ["reg:linear", "reg:gamma", "reg:tweedie"]}
    
    model_name = "xgbr"

    metrics_dict["XGBoost Regressor"] = regressionModel(model_dir, model_name,
                                                        xgbr, x_train, y_train, x_eval, y_eval, score, train_dates, kf, visualization, **params)

    return metrics_dict

def main():

    full_df = pd.read_csv("../../../../Data_thesis/Full_Datasets/Full.csv")
    size = 0.8
    stations = ["Nieuwmarkt", "Nieuwezijds Kolk",
                "Dam", "Spui", "Centraal Station"]
    model_dir = "../../../../Data_thesis/Models/"

    regression_dict = regressionConstruction(full_df, size, stations, model_dir)

    print(regression_dict)


if __name__ == '__main__':
	main()
