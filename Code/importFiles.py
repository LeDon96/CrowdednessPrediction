#Imports
import json
import pandas

def importFile(path):
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