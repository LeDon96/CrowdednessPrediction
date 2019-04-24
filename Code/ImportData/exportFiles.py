#Imports
import pandas as pd

def exportAsCSV(df, path):
    """
    Save DataFrame to CSV file.

    - Input
        - DataFrame
        - path to desired file location
    - Output: CSV file
    """

    #Converts DF to CSV
    df.to_csv(path, index=False)
