import Regression as reg
import Classification as clas
from TrainTestSplit import trainTestSplit, classCrowdednessCounts
import pandas as pd
from sklearn.model_selection import KFold
import pickle

#Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

def regressionModels(full_df, size, stations, model_dir, kf):

    score = "r2"
    metrics_dict = {}
    visualization = True
    cycles = 10

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

    params = {"learning_rate": [0.05, 0.1, 0.15, 0.2, 0.25],
              "n_estimators": list(range(100, 400,25)),
              "booster": ["gbtree"],
              "objective": ["reg:linear", "reg:gamma", "reg:tweedie"]}
    
    model_name = "xgbr"

    metrics_dict["XGBoost Regressor"] = reg.modelConstruction(model_dir, model_name,
                                                        xgbr, x_train, y_train, x_eval, y_eval, score, train_dates, kf, cycles, visualization, **params)

    return metrics_dict

def classificationConstruction(class_df, size, stations, kf, model_dir):

    labels = [1, 2, 3, 4]
    metrics_dict = {}
    score = "f1_weighted"
    cycles = 10
    visualization = True

    x_train, y_train, x_eval, y_eval, train_dates = trainTestSplit(
        class_df, size, stations)

    #Baseline
    base = DummyClassifier()

    params = {"strategy": ["stratified", "most_frequent", "prior", "uniform"]}

    model_name = "base"

    metrics_dict["Classification Baseline"] = clas.modelConstruction(
        model_dir, model_name, base, labels, x_train, y_train, x_eval, y_eval, score, train_dates, kf, cycles, visualization, **params)

    #Random Forrest Classification
    rfc = RandomForestClassifier()

    params = {"n_estimators": list(range(300, 400, 25)),
              "criterion": ["gini", "entropy"],
              "max_features": ["log2", "auto", None],
              "bootstrap": [True],
              "oob_score": [True, False],
              "class_weight": ["balanced", "balanced_subsample", None]}

    model_name = "rfc"

    metrics_dict["Random Forrest Classification"] = clas.modelConstruction(
        model_dir, model_name, rfc, labels, x_train, y_train, x_eval, y_eval, score, train_dates, kf, cycles, visualization, **params)

    #XGBoost Classification
    xgbc = xgb.XGBClassifier()

    params = {"learning_rate": [0.05, 0.1, 0.15, 0.2, 0.25],
              "n_estimators": list(range(100, 300, 25)),
              "booster": ["gbtree"],
              "objective": ["multi:softmax", "multi:softprob"]}

    model_name = "xgbc"

    metrics_dict["XGBoost Classification"] = clas.modelConstruction(
        model_dir, model_name, xgbc, labels, x_train, y_train, x_eval, y_eval, score, train_dates, kf, cycles, visualization, **params)

    return metrics_dict

def main():

    #Variables
    full_df = pd.read_csv("../../../../Data_thesis/Full_Datasets/Full.csv")
    size = 0.8
    stations = ["Nieuwmarkt", "Nieuwezijds Kolk",
                "Dam", "Spui", "Centraal Station"]
    model_dir = "../../../../Data_thesis/Models/"
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    reg_df_path = "../../../../Data_thesis/Full_Datasets/RegModelResults.csv"
    clas_df_path = "../../../../Data_thesis/Full_Datasets/ClasModelResults.csv"


    #Regression
    regression_dict = regressionModels(full_df, size, stations, model_dir, kf)
    reg_df = pd.DataFrame.from_dict(regression_dict, orient="index")
    reg_df.to_csv(reg_df_path, index=True)

    #Classification
    class_df = classCrowdednessCounts(full_df)
    classification_dict = classificationConstruction(class_df, size, stations, kf, model_dir)
    clas_df = pd.DataFrame.from_dict(classification_dict, orient="index")  
    clas_df.to_csv(clas_df_path, index=True)


if __name__ == '__main__':
	main()
