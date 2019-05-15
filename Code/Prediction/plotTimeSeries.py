from bubbly.bubbly import bubbleplot
import plotly 

def plotTimeSeries(df, date):

    #Variables
    x_column = 'SensorLongitude'
    y_column = 'SensorLatitude'
    bubble_column = 'Sensor'
    time_column = 'Hour'
    size_column = 'CrowdednessCount'
    str_date = pd.Timestamp.strftime(date, format="%Y-%m-%d")

    #Initialize grid
    grid = pd.DataFrame()

    figure = bubbleplot(dataset=df, x_column=x_column, y_column=y_column,
                        bubble_column=bubble_column, size_column=size_column, time_column=time_column, color_column=bubble_column,
                        x_title="Sensor Longitude", y_title="Sensor Latitude", title='Crowdedness Counts Amsterdam - ' + str_date,
                        x_logscale=False, scale_bubble=3, height=650, x_range=[min(df[x_column])-0.001, max(df[x_column])+0.001],
                        y_range=[min(df[y_column])-0.001, max(df[y_column])+0.001])

    plotly.offline.plot(figure, filename="../../../Data_thesis/Full_Datasets/Plots/{0}_plot.html".format(str_date),
                        auto_open=False)
