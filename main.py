import pandas as pd 
import numpy as np 
import json 
import re 
import os
from progressbar import ProgressBar, Timer, AdaptiveETA, Bar, Percentage

#Import own functions
from Code.ImportData.constructFullDataset import constructDF
from Code.Models.models import models
from Code.Prediction.Prediction import prediction
from Code.Prediction.minMaxCoordinates import minMaxCoordinates

#Dict with input file paths

def setUp():
    with open("ParamSettings/InputFilePaths.txt", "r") as f:
        input_dict = eval(f.read())

    with open("ParamSettings/OutputFilePaths.txt", "r") as f:
        output_dict = eval(f.read())

    with open("ParamSettings/HParams.txt", "r") as f:
        params_dict = eval(f.read())

    with open("ParamSettings/ModelParams.txt", "r") as f:
        models_dict = eval(f.read())
    
    with open("ParamSettings/PredParams.txt", "r") as f:
        pred_dict = eval(f.read())

    if os.path.isdir("Output") == False:
        os.makedirs("Output/Datasets")
        os.makedirs("Output/Models")
        os.makedirs("Output/Plots")

    return input_dict, output_dict, params_dict, models_dict, pred_dict

def main():

    max_value = 4
    widgets = [Percentage(), Bar(), " ", Timer(), ", ",
               AdaptiveETA()]
    pbar = ProgressBar(widgets=widgets, maxval=max_value).start()

    
    input_dict, output_dict, params_dict, models_dict, pred_dict = setUp()

    if params_dict["make_fullDF"]:
        max_value += 5

    if params_dict["make_models"]:
        max_value += 2 + \
            len(params_dict["reg_models"]) + len(params_dict["clas_models"])

    for i in range(max_value):

        if params_dict["make_fullDF"]:
            constructDF(input_dict, output_dict, params_dict, pbar, i)

            full_df = pd.read_csv(output_dict["full_df"])

            params_dict["lon_max"], params_dict["lon_min"], params_dict["lat_max"], params_dict["lat_min"] = minMaxCoordinates(
                full_df)

            with open("ParamSettings/HParams.txt", "w") as f:
                f.write(str(params_dict))

        if params_dict["make_models"]:
            models(output_dict, params_dict, models_dict, pbar, i)

        prediction(output_dict, params_dict, pred_dict, pbar, i)
    pbar.finish()

if __name__ == '__main__':
    main()
