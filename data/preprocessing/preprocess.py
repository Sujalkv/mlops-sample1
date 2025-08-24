import argparse
import pandas as pd
import numpy as np
import os
import boto3
from sklearn.model_selection import train_test_split

def preprocess(input_data_path, output_data_path):
    # Load data
    print(f"Loading data from {input_data_path}")
    data = pd.read_csv(f"{input_data_path}/raw_data.csv")
    
    # Perform preprocessing steps
    # Example: handle missing values, feature engineering, etc.
    data = data.fillna(data.mean())
    
    # Split data into train, validation, test sets
    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
    train_data, validation_data = train_test_split(train_data, test_size=0.2, random_state=42)
    
    # Save processed datasets
    train_data.to_csv(f"{output_data_path}/train.csv", index=False)
    validation_data.to_csv(f"{output_data_path}/validation.csv", index=False)
    test_data.to_csv(f"{output_data_path}/test.csv", index=False)
    
    print("Preprocessing completed successfully")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-data-path", type=str, required=True)
    parser.add_argument("--output-data-path", type=str, required=True)
    args = parser.parse_args()
    
    preprocess(args.input_data_path, args.output_data_path)
