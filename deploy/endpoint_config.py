import argparse
import boto3
import json
import os
import time

def deploy_model(model_data_url, model_name, role_arn, instance_type="ml.m5.large"):
    """
    Deploy a model to SageMaker endpoint
    """
    sagemaker_client = boto3.client('sagemaker')
    
    # Create model in SageMaker
    print(f"Creating model: {model_name}")
    create_model_response = sagemaker_client.create_model(
        ModelName=model_name,
        PrimaryContainer={
            'Image': '683313688378.dkr.ecr.us-east-1.amazonaws.com/sagemaker-scikit-learn:0.23-1-cpu-py3',
            'ModelDataUrl': model_data_url,
        },
        ExecutionRoleArn=role_arn
    )
    
    # Create endpoint configuration
    endpoint_config_name = f"{model_name}-config-{int(time.time())}"
    print(f"Creating endpoint configuration: {endpoint_config_name}")
    create_endpoint_config_response = sagemaker_client.create_endpoint_config(
        EndpointConfigName=endpoint_config_name,
        ProductionVariants=[
            {
                'VariantName': 'AllTraffic',
                'ModelName': model_name,
                'InitialInstanceCount': 1,
                'InstanceType': instance_type
            }
        ]
    )
    
    # Create endpoint
    endpoint_name = f"{model_name}-endpoint-{int(time.time())}"
    print(f"Creating endpoint: {endpoint_name}")
    create_endpoint_response = sagemaker_client.create_endpoint(
        EndpointName=endpoint_name,
        EndpointConfigName=endpoint_config_name
    )
    
    print(f"Endpoint creation initiated: {endpoint_name}")
    print("Waiting for endpoint to be in service...")
    
    # Write endpoint info to file
    endpoint_info = {
        "endpoint_name": endpoint_name,
        "model_name": model_name,
        "endpoint_config_name": endpoint_config_name
    }
    
    with open("endpoint_info.json", "w") as f:
        json.dump(endpoint_info, f)
    
    return endpoint_name

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-data-url", type=str, required=True)
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--role-arn", type=str, required=True)
    parser.add_argument("--instance-type", type=str, default="ml.m5.large")
    args = parser.parse_args()
    
    deploy_model(args.model_data_url, args.model_name, args.role_arn, args.instance_type)
