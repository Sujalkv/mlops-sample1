# mlops-sample1
mlops-sample1

MLOps Pipeline with AWS Native Services
This repository contains an end-to-end MLOps pipeline implementation using AWS native services including GitHub, CodePipeline, and SageMaker. The pipeline automates the process of data preprocessing, model training, evaluation, and deployment.

Table of Contents
Architecture Overview
Prerequisites
Repository Structure
Setup Instructions
Pipeline Execution
Manual Steps
Monitoring and Maintenance
Troubleshooting
Architecture Overview
This MLOps pipeline consists of:

Source Control: GitHub repository
CI/CD: AWS CodePipeline with CodeBuild
ML Training & Deployment: AWS SageMaker
Storage: S3 buckets for data and models
Model Registry: SageMaker Model Registry
The pipeline has three main stages:

Data Processing: Preprocess raw data and store it in S3
Model Training: Train ML model using SageMaker and evaluate its performance
Model Deployment: Deploy the model to a SageMaker endpoint
Prerequisites
AWS Account with appropriate permissions
GitHub account
AWS CLI installed and configured
Basic knowledge of Python, AWS services, and machine learning
Repository Structure
mlops-sample1/
├── data/
│   └── preprocessing/
│       └── preprocess.py
├── model/
│   ├── train.py
│   └── evaluate.py
├── deploy/
│   └── endpoint_config.py
├── buildspec_preprocess.yml
├── buildspec_train.yml
├── buildspec_deploy.yml
├── pipeline.yml
└── README.md
Setup Instructions
1. Clone the Repository
git clone https://github.com/yourusername/mlops-sample1.git
cd mlops-sample1
2. Create GitHub Personal Access Token
Log in to your GitHub account
Go to Settings > Developer settings > Personal access tokens
Click "Generate new token" (classic)
Give it a name and select the following scopes:
repo (Full control of private repositories)
admin:repo_hook (Full control of repository hooks)
Copy the generated token (you'll need it for CloudFormation)
3. Deploy CloudFormation Stack
Open the AWS Management Console and navigate to CloudFormation
Click "Create stack" > "With new resources"
Upload the pipeline.yml file
Enter stack details:
Stack name: mlops-pipeline
GitHubOwner: Your GitHub username
GitHubRepo: Your repository name
GitHubBranch: main (or your default branch)
GitHubToken: The token you created
Follow the prompts to create the stack
4. Upload Initial Data
After the CloudFormation stack is created, you'll need to upload the sample dataset:

# Replace with your actual bucket name from CloudFormation outputs
aws s3 cp raw_data.csv s3://mlops-pipeline-data-ACCOUNT_ID/raw_data.csv
Pipeline Execution
The pipeline will automatically execute when changes are pushed to your GitHub repository. You can also manually trigger it:

Go to the AWS CodePipeline console
Select your pipeline
Click "Release change"
Click "Release"
Manual Steps
Creating and Uploading Sample Data
The pipeline requires a dataset to process. We've provided a sample customer churn dataset:

Create a file named raw_data.csv with the sample data (see below)
Upload it to your data bucket:
aws s3 cp raw_data.csv s3://mlops-pipeline-data-ACCOUNT_ID/raw_data.csv
Sample Dataset Format
The sample dataset is for customer churn prediction with the following columns:

customer_id: Unique identifier
age: Customer's age
tenure_months: Number of months as customer
monthly_charges: Monthly charges in dollars
total_charges: Total charges to date
contract_type: Type of contract (0=Month-to-month, 1=One year, 2=Two year)
payment_method: Payment method (0-3)
Various service features (0=No, 1=Yes)
target: Whether the customer churned (0=No, 1=Yes)
Manual Model Upload (if needed)
If you need to upload a pre-trained model:

# Replace with your actual bucket name from CloudFormation outputs
aws s3 cp model.pkl s3://mlops-pipeline-model-ACCOUNT_ID/model/model.pkl
Monitoring and Maintenance
Checking Pipeline Status
Go to the AWS CodePipeline console
Select your pipeline
View the current status and history
Accessing SageMaker Endpoints
After successful deployment:

Go to the AWS SageMaker console
Navigate to "Endpoints"
Find your endpoint (format: ml-model-YYYY-MM-DD-HH-MM-SS-endpoint-TIMESTAMP)
Testing the Deployed Model
import boto3
import json
import pandas as pd

# Get endpoint name from SageMaker console or from endpoint_info.json
endpoint_name = "your-endpoint-name"

# Create sample input data
sample_data = {
    "age": [42],
    "tenure_months": [24],
    "monthly_charges": [65.5],
    # Add all required features
}

# Convert to CSV format for inference
sample_df = pd.DataFrame(sample_data)
payload = sample_df.to_csv(index=False, header=False)

# Create SageMaker runtime client
runtime = boto3.client('sagemaker-runtime')

# Invoke endpoint
response = runtime.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='text/csv',
    Body=payload
)

# Parse response
result = json.loads(response['Body'].read().decode())
print(f"Prediction: {result}")
Troubleshooting
Common Issues and Solutions
Pipeline Fails at Source Stage
Check GitHub token permissions
Verify webhook setup in GitHub repository settings
Ensure the repository is accessible
Preprocessing Stage Fails
Check if raw_data.csv is uploaded to the correct S3 location
Verify IAM permissions for CodeBuild to access S3
Check buildspec_preprocess.yml for correct paths
Training Stage Fails
Check if processed data exists in S3
Verify that the training script can access the data
Check for Python package dependencies in buildspec_train.yml
Deployment Stage Fails
Ensure model artifacts are correctly saved
Check IAM permissions for SageMaker deployment
Verify endpoint configuration parameters
Logs and Debugging
CodeBuild Logs: Available in AWS CloudWatch Logs
Pipeline Execution History: Available in CodePipeline console
SageMaker Training Jobs: Check SageMaker console for training job logs
SageMaker Endpoints: Monitor endpoint metrics in CloudWatch
Additional Resources
AWS CodePipeline Documentation
AWS SageMaker Documentation
AWS CloudFormation Documentation
This MLOps pipeline demonstrates best practices for automating machine learning workflows on AWS. It provides a foundation that you can extend and customize for your specific ML use cases.
