#Imports
import json
import pandas as pd
import re
import numpy as np

#Import Functions other files
import ImportData.importFiles as im
import ImportData.exportFiles as ex

def DateToDatetime(df, format):
    """
    Convert column to datetime object
    """

    return pd.to_datetime(df, format=format)

def StartEndDate(df):
    """
    Return the start and end date of a given df
    """

    #Return the earliest and latest date in Date column of given dataframe
    return df.min().Date, df.max().Date 

def importData(crowd_df, gvb_df, event_df):
    """
    Import Data from given file location and save as DF. 
    Furhtermore, change date from string to datetime object
    """

    #Format Datetime
    date_format = "%Y-%m-%d"

    #Crowdedness DF
    crowd_df["Date"] = DateToDatetime(crowd_df["Date"], date_format)

    #GVB DF
    gvb_df["Date"] = DateToDatetime(gvb_df["Date"], date_format)

    #Event Df
    event_df["Date"] = DateToDatetime(event_df["Date"], date_format)

    return crowd_df, gvb_df, event_df

def changeStartEndDate(crowd_df, gvb_df, event_df):
    """
    Change the start and end date of the GVB and Event Df's
    """
    
    #Select start and end date
    start_date, end_date = StartEndDate(crowd_df)

    gvb_df = gvb_df[(gvb_df["Date"] >= start_date) & (
        gvb_df["Date"] <= end_date)].reset_index().drop(columns=["index"])

    event_df = event_df[(event_df["Date"] >= start_date) & (
        event_df["Date"] <= end_date)].reset_index().drop(columns=["index"])

    return gvb_df, event_df

def formFullDF(crowd_df, gvb_df, event_df):
    """
    Construct Full DF 
    """

    #Combine DF's
    gvb_crowd_df = pd.merge(gvb_df, crowd_df, on=["Date", "Hour"], how="outer")
    full_df = pd.merge(gvb_crowd_df, event_df, on=["Date"], how="outer")

    #Sort keys on date
    full_df = full_df.sort_values(
        by=["Date"]).reset_index().drop(columns=["index"])

    #Fill NaN values with 0.0
    full_df = full_df.fillna(0.0)

    #Add columns for the cos and sin of month, day and year
    full_df = full_df.assign(Year=0, month_sin=0, month_cos=0,
                            day_sin=0, day_cos=0, hour_sin=0, hour_cos=0)

    #Transform DF to Dict
    time_dict = full_df.to_dict("index")

    #Transform Date to seperate year, month, day and hour. And transform month, day, hour to cos/sin to make it circular
    for k, v in time_dict.items():
        v["Year"] = v["Date"].year

        v["month_sin"] = np.sin(2 * np.pi * v["Date"].month / 12)
        v["month_cos"] = np.cos(2 * np.pi * v["Date"].month / 12)

        v["day_sin"] = np.sin(2 * np.pi * v["Date"].day / 365)
        v["day_cos"] = np.cos(2 * np.pi * v["Date"].day / 365)

        v["hour_sin"] = np.sin(2 * np.pi * v["Hour"] / 2400)
        v["hour_cos"] = np.cos(2 * np.pi * v["Hour"] / 2400)

    #Transform dict back to DF
    return pd.DataFrame.from_dict(time_dict, orient="index").reset_index()


def CombineDF(crowd_df, gvb_df, event_df):

    #Import the needed CSV files
    crowd_df, gvb_df, event_df = importData(crowd_df, gvb_df, event_df)

    #Change start and end date of DF's
    gvb_df, event_df = changeStartEndDate(crowd_df, gvb_df, event_df)

    #Form full DF
    full_df = formFullDF(crowd_df, gvb_df, event_df)

    return full_df
