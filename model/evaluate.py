import argparse
import os
import pandas as pd
import numpy as np
import boto3
import json
import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate(model_path, test_data_path, output_dir):
    print("Starting model evaluation")
    
    # Load model
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Load test data
    test_data = pd.read_csv(f"{test_data_path}/test.csv")
    X_test = test_data.drop('target', axis=1)
    y_test = test_data['target']
    
    # Make predictions
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    print(f"Test metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # Save metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    
    with open(os.path.join(output_dir, "evaluation.json"), "w") as f:
        json.dump(metrics, f)
    
    print(f"Evaluation metrics saved to {output_dir}/evaluation.json")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True)
    parser.add_argument("--test-data-path", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    
    evaluate(args.model_path, args.test_data_path, args.output_dir)
