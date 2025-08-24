import argparse
import os
import pandas as pd
import numpy as np
import boto3
import json
import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def train(train_data_path, validation_data_path, model_dir):
    print("Starting model training")
    
    # Load training and validation data
    train_data = pd.read_csv(f"{train_data_path}/train.csv")
    validation_data = pd.read_csv(f"{validation_data_path}/validation.csv")
    
    # Separate features and target
    X_train = train_data.drop('target', axis=1)
    y_train = train_data['target']
    X_val = validation_data.drop('target', axis=1)
    y_val = validation_data['target']
    
    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate on validation set
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    precision = precision_score(y_val, y_pred, average='weighted')
    recall = recall_score(y_val, y_pred, average='weighted')
    f1 = f1_score(y_val, y_pred, average='weighted')
    
    print(f"Validation metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}")
    
    # Save model
    model_path = os.path.join(model_dir, "model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Save metrics
    metrics = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1)
    }
    
    with open(os.path.join(model_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f)
    
    print(f"Model saved to {model_path}")
    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-data-path", type=str, required=True)
    parser.add_argument("--validation-data-path", type=str, required=True)
    parser.add_argument("--model-dir", type=str, required=True)
    args = parser.parse_args()
    
    train(args.train_data_path, args.validation_data_path, args.model_dir)
