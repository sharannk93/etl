import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np
from process import load_data, prepare_data, store_result
from prefect import flow, task

# Set the paths for data loading and result storage
data_path = r"C:\Users\shara"
result_path = r"D:\test\result.csv"

@task
# Load the data
def load(data_path):
    raw_df, fif = load_data(data_path)
    return raw_df, fif

@task
# Prepare the data
def prepare(raw_df,fif):
    filled_df = prepare_data(raw_df, fif)
    return filled_df

@flow (name = "Flow info", log_prints=True)
# Store the final result
def store():
    # Set the paths for data loading and result storage
    data_path = r"C:\Users\shara"
    result_path = r"D:\test\result.csv"
    raw_df,fif = load(data_path)
    filled_df = prepare(raw_df,fif)
    store_result(filled_df, result_path)

if __name__ == "__main__":
    store()
