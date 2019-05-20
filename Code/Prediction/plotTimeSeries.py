from Code.bubbly.bubbly import bubbleplot
import plotly 
import pandas as pd

def plotTimeSeries(df, date, output_dict):
    """
    This function constructs a crowdedness counts plot for a single date, containing data for all given sensors

    Parameters:
    - df (df): contains all prediction data
    - date (Timestamp): given date
    - output_dict (dict): all paths for output files

    Returns
    - a bubbleplot, saved at specified directory
    """

    #Variables
    x_column = 'SensorLongitude'
    y_column = 'SensorLatitude'
    bubble_column = 'Sensor'
    time_column = 'Hour'
    size_column = 'CrowdednessCount'
    str_date = pd.Timestamp.strftime(date, format="%Y-%m-%d")

    #Initialize grid
    grid = pd.DataFrame()

    #Construct bubble plot
    figure = bubbleplot(dataset=df, x_column=x_column, y_column=y_column,
                        bubble_column=bubble_column, size_column=size_column, time_column=time_column, color_column=bubble_column,
                        x_title="Sensor Longitude", y_title="Sensor Latitude", title='Crowdedness Counts Amsterdam - ' + str_date,
                        x_logscale=False, scale_bubble=3, height=650, x_range=[min(df[x_column])-0.001, max(df[x_column])+0.001],
                        y_range=[min(df[y_column])-0.001, max(df[y_column])+0.001])

    #Save at specifed location
    plotly.offline.plot(figure, filename=output_dict["plots"] + "{0}_plot.html".format(str_date),
                        auto_open=False)
