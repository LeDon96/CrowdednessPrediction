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

def regressionModels(output_dict, params_dict, models_dict, kf, full_df):
    """
    This function constructs the Regression models

    Parameters:
    - output_dict (dict): all paths of where output files should be saved
    - params_dict (dict): all general hyperparameters that can be changed by user
    - models_dict (dict): all parameters for the models
    - kf (model): KFold split dataset n times
    - full_df (df): Full dataset with all data

    Returns:
    - Save models to pickle
    - Save result models to CSV
    """

    #Dict to save model results in
    metrics_dict = {}

    #If remove sensor, split the data on sensors
    if params_dict["remove_sensor"]:
        x_train = full_df[full_df["Sensor"] != params_dict["sensor_to_remove"]].drop(
            columns=["CrowdednessCount"])
        y_train = full_df[full_df["Sensor"] !=
                          params_dict["sensor_to_remove"]][["Date", "CrowdednessCount"]]

        train_dates = full_df["Date"].unique()

        x_eval = full_df[full_df["Sensor"] == params_dict["sensor_to_remove"]].drop(
            columns=["Date", "CrowdednessCount"])
        y_eval = full_df[full_df["Sensor"] ==
                          params_dict["sensor_to_remove"]]["CrowdednessCount"]
    
    #Split date into train and evaluation
    x_train, y_train, x_eval, y_eval, train_dates = trainTestSplit(
        full_df, models_dict["trainTest"]["size"])

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
            train_dates, kf, models_dict[name]["cycles"], models_dict[name]["params"], models_dict["KFold"]["size"],
            params_dict["remove_sensor"])

    #Save model results in dict
    df = pd.DataFrame.from_dict(metrics_dict, orient="index")

    if params_dict["remove_sensor"]:
        df.to_csv(output_dict["gen_reg_metrics"], index=True)
    else:
        df.to_csv(output_dict["reg_metrics"], index=True)

def classificationModels(output_dict, params_dict, models_dict, kf, full_df):
    """
    This function constructs the Classification models

    Parameters:
    - output_dict (dict): all paths of where output files should be saved
    - params_dict (dict): all general hyperparameters that can be changed by user
    - models_dict (dict): all parameters for the models
    - kf (model): KFold split dataset n times
    - full_df (df): Full dataset with all data

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

    #If remove sensor, split the data on sensors
    if params_dict["remove_sensor"]:
        x_train = class_df[class_df["Sensor"] != params_dict["sensor_to_remove"]].drop(
            columns=["CrowdednessCount"])
        y_train = class_df[class_df["Sensor"] !=
                          params_dict["sensor_to_remove"]][["Date", "CrowdednessCount"]]

        train_dates = class_df["Date"].unique()

        x_eval = class_df[class_df["Sensor"] == params_dict["sensor_to_remove"]].drop(
            columns=["Date", "CrowdednessCount"])
        y_eval = class_df[class_df["Sensor"] ==
                         params_dict["sensor_to_remove"]]["CrowdednessCount"]

    #Split date into train and evaluation
    x_train, y_train, x_eval, y_eval, train_dates = trainTestSplit(
        class_df, models_dict["trainTest"]["size"])

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
            train_dates, kf, models_dict[name]["cycles"], models_dict[name]["params"], models_dict["KFold"]["size"],
            params_dict["remove_sensor"])

    #Save model results in dict
    df = pd.DataFrame.from_dict(metrics_dict, orient="index")

    if params_dict["remove_sensor"]:
        df.to_csv(output_dict["gen_clas_metrics"], index=True)
    else:
        df.to_csv(output_dict["clas_metrics"], index=True)


def models(output_dict, params_dict, models_dict, pred_dict):
    """
    This function calls on functions to construct all the models

    Parameters:
    - output_dict (dict): all paths of where output files should be saved
    - params_dict (dict): all general hyperparameters that can be changed by user
    - models_dict (dict): all parameters for the models
    - pred_dict (dict): all parameters for the prediction

    Returns:
    - Saves constructed models
    - Saves results constructed models
    """

    #Splits given data n times
    kf = KFold(n_splits=models_dict["KFold"]["size"],
               shuffle=models_dict["KFold"]["shuffle"], random_state=42)

    #Import Dataset
    full_df = pd.read_csv(output_dict["full_df"])

    #Drop the unscaled station coordinates
    for station in params_dict["stations"]:
        full_df.drop(columns={station + " Lon", station + " Lat",
                         station + " passengers"}, inplace=True)

    if not params_dict["remove_sensor"]:
        #Remove dates dataset used in prediction
        split_date = pd.to_datetime(pred_dict["start_date"], format="%Y-%m-%d")
        full_df["Date"] = pd.to_datetime(
            full_df["Date"], format="%Y-%m-%d")
        full_df = full_df[full_df["Date"] <= split_date].reset_index().drop(columns=[
            "index"])

    #Construct models
    regressionModels(output_dict, params_dict, models_dict, kf, full_df)
    classificationModels(output_dict, params_dict, models_dict, kf, full_df)