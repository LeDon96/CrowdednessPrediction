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
    """
    This function constructs the Regression models

    Parameters:
    - output_dict (dict): all paths of where output files should be saved
    - params_dict (dict): all general hyperparameters that can be changed by user
    - models_dict (dict): all parameters for the models
    - kf (model): KFold split dataset n times
    - full_df (df): Full dataset with all data
    - pbar: Progress bar 
    - i: current number of iteration the progress is at

    Returns:
    - Save models to pickle
    - Save result models to CSV
    """

    #Dict to save model results in
    metrics_dict = {}

    #Split date into train and evaluation
    x_train, y_train, x_eval, y_eval, train_dates = trainTestSplit(
        full_df, models_dict["trainTest"]["size"], params_dict["stations"])

    #Loop over all needed model ID's and select the model based on the ID
    for name in params_dict["reg_models"]:
        if name == "lr":
            model = LinearRegression()
        elif name == "rfg":
            model = RandomForestRegressor()
        elif name == "xgbr":
            model = XGBRegressor()

        #Construct the model and save the model results
        metrics_dict[name] = reg.modelConstruction(
            output_dict["models"], output_dict["plots"], name, model, x_train, y_train, x_eval, y_eval, models_dict[name]["score"],
            train_dates, kf, models_dict[name]["cycles"], models_dict[name]["visualization"], models_dict[name]["params"], models_dict["KFold"]["size"],
            models_dict["saveResults"], output_dict["predictions"])

        #Advanced iteration progressbar
        pbar.update(i+1)

    #Save model results in dict
    df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    df.to_csv(output_dict["reg_metrics"], index=True)

    #Advanced iteration progressbar
    pbar.update(i+1)


def classificationModels(output_dict, params_dict, models_dict, kf, full_df, pbar, i):
    """
    This function constructs the Classification models

    Parameters:
    - output_dict (dict): all paths of where output files should be saved
    - params_dict (dict): all general hyperparameters that can be changed by user
    - models_dict (dict): all parameters for the models
    - kf (model): KFold split dataset n times
    - full_df (df): Full dataset with all data
    - pbar: Progress bar 
    - i: current number of iteration the progress is at

    Returns:
    - Save models to pickle
    - Save result models to CSV
    """

    #Dict to save model results in
    metrics_dict = {}

    #Label of all the classes
    labels = [1, 2, 3, 4]

    #Convert the numerical crowdednessCounts to class labels
    class_df = classCrowdednessCounts(full_df)

    #Split date into train and evaluation
    x_train, y_train, x_eval, y_eval, train_dates = trainTestSplit(
        class_df, models_dict["trainTest"]["size"], params_dict["stations"])

    #Loop over all needed model ID's and select the model based on the ID
    for name in params_dict["clas_models"]:
        if name == "dc":
            model = DummyClassifier()
        elif name == "rfc":
            model = RandomForestClassifier()
        elif name == "xgbc":
            model = XGBClassifier()

        #Construct the model and save the model results
        metrics_dict[name] = clas.modelConstruction(
            output_dict["models"], output_dict["plots"], name, model, labels, x_train, y_train, x_eval, y_eval, models_dict[name]["score"],
            train_dates, kf, models_dict[name]["cycles"], models_dict[name]["visualization"], models_dict[name]["params"], models_dict["KFold"]["size"],
            models_dict["saveResults"], output_dict["predictions"])

        #Advanced iteration progressbar
        pbar.update(i+1)

    #Save model results in dict
    df = pd.DataFrame.from_dict(metrics_dict, orient="index")
    df.to_csv(output_dict["clas_metrics"], index=True)

    #Advanced iteration progressbar
    pbar.update(i+1)


def models(output_dict, params_dict, models_dict, pred_dict, pbar, i):
    """
    This function calls on functions to construct all the models

    Parameters:
    - output_dict (dict): all paths of where output files should be saved
    - params_dict (dict): all general hyperparameters that can be changed by user
    - models_dict (dict): all parameters for the models
    - pbar: Progress bar 
    - i: current number of iteration the progress is at

    Returns:
    - Saves constructed models
    - Saves results constructed models
    """

    #Splits given data n times
    kf = KFold(n_splits=models_dict["KFold"]["size"],
               shuffle=models_dict["KFold"]["shuffle"], random_state=42)

    #Import Dataset
    full_df = pd.read_csv(output_dict["full_df"])

    #Remove dates dataset used in prediction
    split_date = pd.to_datetime(pred_dict["start_date"], format="%Y-%m-%d")
    full_df["Date"] = pd.to_datetime(
        full_df["Date"], format="%Y-%m-%d")
    full_df = full_df[full_df["Date"] <= split_date].reset_index().drop(columns=[
        "index"])

    #Construct models
    regressionModels(output_dict, params_dict, models_dict, kf, full_df, pbar, i)
    classificationModels(output_dict, params_dict, models_dict, kf, full_df, pbar, i)
