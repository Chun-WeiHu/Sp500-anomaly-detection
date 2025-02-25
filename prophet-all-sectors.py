from prophet import Prophet
import numpy as np
import pandas as pd
import sys
#import ax
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly, plot_components_plotly
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

all_performance = []

for i in range(1, len(sys.argv)):
    dataframe = pd.read_csv(sys.argv[i])

    dataframe['ds'] = pd.to_datetime(dataframe['ds'], format='%Y-%m-%d')

    dataframe = dataframe.loc[(dataframe['ds'] >= '1990-01-01')
                         & (dataframe['ds'] < '2030-01-01')]
    
    dataframe['region'] = (sys.argv[i].split('_')[-1]).split('.csv')[0]
#    print(dataframe)
    dataframe.head()

    model = Prophet(interval_width=0.95, yearly_seasonality=True, weekly_seasonality=True)
    model.fit(dataframe)

    forecast = model.predict(dataframe)

    future = model.make_future_dataframe(periods = 1826)

    future_pts = model.predict(future)

    predicted = future_pts.loc[(future_pts['ds'] >= '2020-04-01')]

#    print(future_pts)
    # Save the interactive plot to a variable
    plot = plot_plotly(model, forecast)

    plot2 = plot_components_plotly(model, forecast)

    # Display the interactive plot
    plot.show()
    plot2.show()

    # Merge actual and predicted values
    performance = pd.merge(dataframe, forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']], on='ds')

    print("Number of NaN values in the DataFrame:")
    print(performance.isna().sum())
    performance = performance.dropna()

    # Check MAE value
    performance_MAE = mean_absolute_error(performance['y'], performance['yhat'])
    print(f'The mean_absolute_error for the model is {performance_MAE}')

    # Check MAPE value
    performance_MAPE = mean_absolute_percentage_error(performance['y'], performance['yhat'])
    print(f'The mean_absolute_percentage_error for the model is {performance_MAPE*100}%')
    
    if i == 1:
        all_performance = performance
    else:
        all_performance = pd.concat([all_performance, performance])
#    performance['anomaly'] = performance.apply(lambda rows: 1 if ((rows.y<rows.yhat_lower)|(rows.y>rows.yhat_upper)) else 0, axis = 1)
#    scatter_plot = sns.scatterplot(x='ds', y='y', data=performance, hue='anomaly')
#    plt.show()
line_plot = sns.lineplot(x='ds', y='y', data=all_performance, hue='region')
#    line_plot = sns.lineplot(x='ds', y='yhat', data=performance, color='red')
plt.show()

