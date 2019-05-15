import pandas as pd 
import numpy as np 
import json 
import re 
import os

#Import own functions
from Code.ImportData.constructFullDataset import constructDF
from Code.Models.models import models

#Dict with input file paths

def setUp():
    with open('InputFilePaths.txt', 'r') as f:
        input_dict = eval(f.read())

    with open('OutputFilePaths.txt', 'r') as f:
        output_dict = eval(f.read())

    with open('HPSettings.txt', 'r') as f:
        params_dict = eval(f.read())

    with open('ModelSettings.txt', 'r') as f:
        models_dict = eval(f.read())

    if os.path.isdir("Output") == False:
        os.makedirs("Output/Datasets")
        os.makedirs("Output/Models")
        os.makedirs("Output/Plots")

    return input_dict, output_dict, params_dict, models_dict

def main():

    input_dict, output_dict, params_dict, models_dict = setUp()

    if params_dict["make_fullDF"]:
        constructDF(input_dict, output_dict, params_dict)

    if params_dict["make_models"]:
        models(output_dict, params_dict, models_dict)

if __name__ == '__main__':
    main()
