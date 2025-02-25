import os
import pandas as pd
import sys

args = sys.argv

input_folder = sys.argv[1]

dataframes = []

for filename in os.listdir(input_folder):
    if filename.endswith('.csv'):
        input_file = os.path.join(input_folder, filename)
        # Read the CSV file into a pandas DataFrame
        dataframe = pd.read_csv(input_file)
        dataframe['ds'] = pd.to_datetime(dataframe['ds'], format='%Y-%m-%d')
        
        dataframe = dataframe.loc[(dataframe['ds'] >= '1990-01-01') & (dataframe['ds'] < '2030-01-01')]
        dataframe['y'] = dataframe['y'] /dataframe['y'].abs().max()
        dataframes.append(dataframe)
        print("Added file: ", filename, "to dataframe list")

combined_dfs = pd.concat(dataframes, ignore_index=True)
average_per_date = combined_dfs.groupby('ds')['y'].mean()

average_per_date.to_csv(sys.argv[2], header=True)