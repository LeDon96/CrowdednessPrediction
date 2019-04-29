#Imports
import pandas as pd
import numpy as np

def clasCrowdednessCounts(df):
    """
    Divide the numerical counts of crowdedness into 4 classes. These classes asre based on the quantiles taken 
    over all the values. 
    """

    #Quantile splits
    low_split = df["CrowdednessCount"].quantile(.25)
    mid_split = df["CrowdednessCount"].quantile(.5)
    high_split = df["CrowdednessCount"].quantile(.75)

    clas_dict = df.to_dict("index")

    for k, v in clas_dict.items():

        if v["CrowdednessCount"] < low_split:
            v["CrowdednessCount"] = 1
        elif v["CrowdednessCount"] >= low_split and v["CrowdednessCount"] < mid_split:
            v["CrowdednessCount"] = 2
        elif v["CrowdednessCount"] >= mid_split and v["CrowdednessCount"] < high_split:
            v["CrowdednessCount"] = 3
        elif v["CrowdednessCount"] >= high_split:
            v["CrowdednessCount"] = 4
        else:
            print(k, " has class error as it fits in none")

    df = pd.DataFrame.from_dict(clas_dict, orient="index")

    return df

def testSensorLat(df, sensor):
    """
    Select the unique label of the latitude for the given sensor and return this
    """
    return df[df["Sensor"] == sensor].reset_index()["SensorLatitude"][0]


def dateSplit(df, size):
    """
    This function splits the given df based on given dates. 
    
    Input:
        - df: DataFrame that needs to be split into train/test
        - size: Size of the training test between 0 and 1
    """

    dates = df["Date"].values
    np.random.shuffle(dates)
    split = int(dates.size * size)

    train_dates = dates[:split]
    test_dates = dates[split:]

    return train_dates, test_dates


def trainTestSplit(df, size, split_date, sensor):

    TrainTest_df = df[df["Date"] <= split_date].reset_index().drop(
        columns=["index", "Hour", "Sensor", "Year"])
    test_lat = testSensorLat(df, sensor)

    #Split Train/Test based on dates
    train_dates, test_dates = dateSplit(TrainTest_df, size)

    train_df_reg = TrainTest_df[TrainTest_df["Date"].isin(
        train_dates)].reset_index().drop(columns=["Date", "index"])
    test_df_reg = TrainTest_df[TrainTest_df["Date"].isin(
        test_dates)].reset_index().drop(columns=["index"])

    #Train
    x_train_reg = train_df_reg.drop(["CrowdednessCount"], axis=1)
    y_train_reg = train_df_reg["CrowdednessCount"]

    #Test
    x_test_reg = test_df_reg.drop(["CrowdednessCount", "Date"], axis=1)
    y_test_reg = test_df_reg["CrowdednessCount"]

    test_reg_series = test_df_reg[(test_df_reg["SensorLatitude"] == test_lat) &
                                  (test_df_reg["Date"] == test_dates[0])].reset_index()
    x_test_reg_series = test_reg_series.drop(
        ["CrowdednessCount", "Date", "index"], axis=1)
    y_test_reg_series = test_reg_series["CrowdednessCount"]

    feature_labels = x_train_reg.columns.values

    return x_train_reg, y_train_reg, x_test_reg, y_test_reg, x_test_reg_series, y_test_reg_series, feature_labels


def evalSplit(df, split_date, sensor, eval_start_date, eval_end_date):

    Eval_df = df[df["Date"] > split_date].reset_index().drop(
        columns=["index", "Hour", "Sensor", "Year"])
    test_lat = testSensorLat(df, sensor)

    #Timeseries
    x_eval_reg = Eval_df.drop(["CrowdednessCount", "Date"], axis=1)
    y_eval_reg = Eval_df["CrowdednessCount"]

    #Subset timeseries
    sub_series = Eval_df[(Eval_df["SensorLatitude"] == test_lat) &
                         (Eval_df["Date"] >= eval_start_date) &
                         (Eval_df["Date"] <= eval_end_date)].reset_index()

    #Time series
    x_series_reg = sub_series.drop(
        ["Date", "CrowdednessCount", "index"], axis=1)
    y_series_reg = sub_series["CrowdednessCount"]

    return x_eval_reg, y_eval_reg, x_series_reg, y_series_reg

def TrainTestEval(df, split_date, sensor, eval_start_date, eval_end_date, size, clas=False):

    if clas == True:
        df = clasCrowdednessCounts(df)

        x_train, y_train, x_test, y_test, x_test_series, y_test_series, feature_labels = trainTestSplit(
            df, size, split_date, sensor)

        x_eval, y_eval, x_eval_series, y_eval_series = evalSplit(
            df, split_date, sensor, eval_start_date, eval_end_date)

        return x_train, y_train, x_test, y_test, x_test_series, y_test_series, feature_labels, x_eval, y_eval, x_eval_series, y_eval_series
    
    else:
        x_train, y_train, x_test, y_test, x_test_series, y_test_series, feature_labels = trainTestSplit(
            df, size, split_date, sensor)

        x_eval, y_eval, x_eval_series, y_eval_series = evalSplit(
            df, split_date, sensor, eval_start_date, eval_end_date)

        return x_train, y_train, x_test, y_test, x_test_series, y_test_series, feature_labels, x_eval, y_eval, x_eval_series, y_eval_series
