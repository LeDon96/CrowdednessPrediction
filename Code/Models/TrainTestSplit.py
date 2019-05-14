import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

def classCrowdednessCounts(df):
    """
    This function divides the numerical counts of crowdedness into 4 classes. These classes asre based on the quantiles taken 
    over all the values. 

    Parameters: 
    - df (df): Where the numerical counts need to be transformed into classes

    Returns: DF with transformed crowdedness classes
    """

    #Variables

    #Quantile splits
    low_split = df["CrowdednessCount"].quantile(.25)
    mid_split = df["CrowdednessCount"].quantile(.5)
    high_split = df["CrowdednessCount"].quantile(.75)

    #################################################################################

    #Dataframe to dict
    clas_dict = df.to_dict("index")

    #Loop over all crowdedness counts
    for k, v in clas_dict.items():

        #If the crowdedness count is below the 25% quantile, class 1 is assigned
        if v["CrowdednessCount"] < low_split:
            v["CrowdednessCount"] = 1

        #If the crowdedness count is above the 25% quantile and below the 50% quantile, class 2 is assigned
        elif v["CrowdednessCount"] >= low_split and v["CrowdednessCount"] < mid_split:
            v["CrowdednessCount"] = 2

        #If the crowdedness count is above the 50% quantile and below the 75% quantile, class 3 is assigned
        elif v["CrowdednessCount"] >= mid_split and v["CrowdednessCount"] < high_split:
            v["CrowdednessCount"] = 3

        #If the crowdedness count is above the 75% quantile, class 4 is assigned
        elif v["CrowdednessCount"] >= high_split:
            v["CrowdednessCount"] = 4
        else:
            print(k, " has class error as it fits in none")

    df = pd.DataFrame.from_dict(clas_dict, orient="index")

    return df

def dateSplit(df, size):
    """
    This function returns the dates the training and test set should be comprised of 
    
    Parameters:
    - df (df): DataFrame that needs to be split into train/test
    - size (float): Size of the training set between 0 and 1

    Returns: train and test dates
    """

    #Find all unique dates
    dates = df["Date"].unique()

    #Split the dates into train and test
    train_dates, eval_dates = train_test_split(
        dates, train_size=size, test_size=1-size, random_state=42)

    return train_dates, eval_dates


def trainTestSplit(df, size, stations):
    """
    This function splits the given df into a train and test set, based on dates

    Parameters:
    - df (df): Data that needs to be split into train and test
    - size (float): Size of the training set
    - stations (list): all stations present in df

    Returns: x_train (df), y_train (df), x_eval (df), y_eval (df), train_dates (list)
    """

    df = df.drop(columns=["Hour", "Sensor", "Year",
                          "SensorLongitude", "SensorLatitude"])

    for station in stations:
        df.drop(columns={station + " Lon", station + " Lat"}, inplace=True)

    #Split Train/Test based on dates
    train_dates, eval_dates = dateSplit(df, size)

    train_df = df[df["Date"].isin(train_dates)].reset_index().drop(
        columns=["index"])
    eval_df = df[df["Date"].isin(eval_dates)].reset_index().drop(
        columns=["index"])

    #Train
    x_train = train_df.drop(["CrowdednessCount"], axis=1)
    y_train = train_df[["Date", "CrowdednessCount"]]

    #Evaluation
    x_eval = eval_df.drop(["Date", "CrowdednessCount"], axis=1)
    y_eval = eval_df["CrowdednessCount"]

    return x_train, y_train, x_eval, y_eval, train_dates