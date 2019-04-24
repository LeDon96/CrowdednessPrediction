#Imports
import json
import pandas as pd

def importJSON(path):
    """
    This function opens the json file and returns the data as JSON object

    - Input: path to file (including filename itself)
    - Output: data from file as JSON object
    """

    #Opens the JSON file
    with open(path) as file_data:

        #Save data as JSON object
        data = json.load(file_data)

    #Return the JSON object
    return data

def importExcel(path):
    """
    This function opens the excel file and returns the data as DataFrame object

    - Input: path to file (including filename itself)
    - Output: data from file as DataFrame
    """

    return pd.read_excel(path)

def importCSV(path, sep=None):
    """
    This function opens the CSV file and returns the data as DataFrame object

    - Input: path to file (including filename itself)
    - Output: data from file as DataFrame
    """

    if sep != None:
        return pd.read_csv(path, sep=sep)
    else:
        return pd.read_csv(path)
