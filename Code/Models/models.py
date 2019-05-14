import Regression as reg
from TrainTestSplit import trainTestSplit
import pandas as pd
from sklearn.model_selection import KFold
import pickle

#Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb

def regressionModels(full_df, size, stations, model_dir, kf):

    score = "r2"
    metrics_dict = {}
    visualization = True
    cycles = 15

    x_train, y_train, x_eval, y_eval, train_dates = trainTestSplit(
        full_df, size, stations)

    #Linear Regression
    lr = LinearRegression()
    params = {"fit_intercept": [True, False],
              "normalize": [True, False],
              "copy_X": [True, False]}

    model_name = "lr"

    metrics_dict["Linear Regression"] = reg.modelConstruction(model_dir, model_name,
        lr, x_train, y_train, x_eval, y_eval, score, train_dates, kf, cycles, visualization, **params)
    
    #Random Forrest Regressor
    rfg = RandomForestRegressor()

    params = {"n_estimators": list(range(300, 400, 25)),
              "criterion": ["mse"],
              "max_features": ["log2", "auto", None],
              "bootstrap": [True]}

    model_name = "rfg"

    metrics_dict["Random Forest Regressor"] = reg.modelConstruction(model_dir, model_name,
                                                        rfg, x_train, y_train, x_eval, y_eval, score, train_dates, kf, cycles, visualization, **params)

    #XGBoost Regressor
    xgbr = xgb.XGBRegressor()

    params = {"learning_rate": list(range(0, 0.30, 0.05)),
              "n_estimators": list(range(100, 400,25)),
              "booster": ["gbtree"],
              "objective": ["reg:linear", "reg:gamma", "reg:tweedie"]}
    
    model_name = "xgbr"

    metrics_dict["XGBoost Regressor"] = reg.modelConstruction(model_dir, model_name,
                                                        xgbr, x_train, y_train, x_eval, y_eval, score, train_dates, kf, cycles, visualization, **params)

    return metrics_dict

def classificationConstruction()

def main():

    full_df = pd.read_csv("../../../../Data_thesis/Full_Datasets/Full.csv")
    size = 0.8
    stations = ["Nieuwmarkt", "Nieuwezijds Kolk",
                "Dam", "Spui", "Centraal Station"]
    model_dir = "../../../../Data_thesis/Models/"
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    regression_dict = regressionConstruction(full_df, size, stations, model_dir, kf)

    print(regression_dict)


if __name__ == '__main__':
	main()
