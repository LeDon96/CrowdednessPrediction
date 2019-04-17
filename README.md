# CrowdednessPrediction

## Abstract
The aim of this research is to construct a model that predicts the crowdedness at selected spots within the city of Amsterdam, in the form of a time series. Historical data will be gathered from localized crowdedness measure points and will be combined with GVB data. The predictions with the given data will be made with both a multivariate regression model and a gradient boosted decision tree. The predicted crowdedness of both models will be visualized in separate time series over a given period of time. The two models will be compared in the evaluation based on the loss. 

## Link Thesis 
- [Overleaf](https://www.overleaf.com/3825517455cpkbjdbgpwmn)

## Contents
- [Log Book](Documents/LogBook.md)
- [Thesis Design](Documents/Thesis_Design_Crowdedness.pdf)

## Code
- [Requirements running code](Code/Requirements.md)

### Jupyter Test code
- [Jupyter Code](Jupyter%20Notebooks)

### Production Code
- [Import Data](Code/ImportData)
    - [Import Crowdedness Data](Code/ImportData/CrowdednessData.py)
    - [Import Event Data](Code/ImportData/EventData.py)
    - [Import File Functions](Code/ImportData/exportFiles.py)
    - [Export Data to File Functions](Code/ImportData/importFiles.py)