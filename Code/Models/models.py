import Code.Models.Regression as reg
import Code.Models.Classification as clas
from Code.Models.TrainTestSplit import trainTestSplit, classCrowdednessCounts
import pandas as pd
from sklearn.model_selection import KFold
import pickle

#Models
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor, XGBClassifier
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier

def regressionModels(output_dict, params_dict, models_dict, kf, full_df, pbar, i):

    metrics_dict = {}

    x_train, y_train, x_eval, y_eval, train_dates = trainTestSplit(
        full_df, models_dict["trainTest"]["size"], params_dict["stations"])

    for name in params_dict["reg_models"]:
        if name == "lr":
            model = LinearRegression()
        elif name == "rfg":
            model = RandomForestRegressor()
        elif name == "xgbr":
            model = XGBRegressor()

        metrics_dict[name] = reg.modelConstruction(
            output_dict["models"], output_dict["plots"], name, model, x_train, y_train, x_eval, y_eval, models_dict[name]["score"],
            train_dates, kf, models_dict[name]["cycles"], models_dict[name]["visualization"], **models_dict[name]["params"])

        pbar.update(i+1)

    df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    df.to_csv(output_dict["reg_metrics"], index=True)

    pbar.update(i+1)


def classificationModels(output_dict, params_dict, models_dict, kf, full_df, pbar, i):

    metrics_dict = {}
    labels = [1, 2, 3, 4]

    class_df = classCrowdednessCounts(full_df)

    x_train, y_train, x_eval, y_eval, train_dates = trainTestSplit(
        class_df, models_dict["trainTest"]["size"], params_dict["stations"])

    for name in params_dict["clas_models"]:

        if name == "dc":
            model = DummyClassifier()
        elif name == "rfc":
            model = RandomForestClassifier()
        elif name == "xgbc":
            model = XGBClassifier()


        metrics_dict[name] = clas.modelConstruction(
            output_dict["models"], output_dict["plots"], name, model, labels, x_train, y_train, x_eval, y_eval, models_dict[name]["score"],
            train_dates, kf, models_dict[name]["cycles"], models_dict[name]["visualization"], **models_dict[name]["params"])

        pbar.update(i+1)

    df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    df.to_csv(output_dict["clas_metrics"], index=True)

    pbar.update(i+1)


def models(output_dict, params_dict, models_dict, pbar, i):

    #Variables
    kf = KFold(n_splits=10, shuffle=True, random_state=42)

    full_df = pd.read_csv(output_dict["full_df"])

    regressionModels(output_dict, params_dict, models_dict, kf, full_df, pbar, i)
    classificationModels(output_dict, params_dict, models_dict, kf, full_df, pbar, i)
