import pandas as pd 
import numpy as np 
import json 
import re 
import os

#Import own functions
# from Code.ImportData.constructFullDataset import constructDF

#Dict with input file paths
input_file = json.load(open("InputFilePaths.json"))
output_file = json.load(open("OutputFilePaths.json"))

if os.path.isdir("Output") == False:
    os.makedirs("Output/Datasets")
    os.makedirs("Output/Models")
    os.makedirs("Output/Plots")

