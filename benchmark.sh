#! /bin/bash
export MLFLOW_S3_ENDPOINT_URL="https://minio.lab.sspcloud.fr"
export MLFLOW_TRACKING_URI="https://user-jpramil-mlflow.user.lab.sspcloud.fr"
export MLFLOW_EXPERIMENT_NAME="torch-fastText-0"

mlflow run ~/work/torch-fastText/ --entry-point=torch \
    --env-manager=local \
    -P remote_server_uri=$MLFLOW_TRACKING_URI \
    -P experiment_name=$MLFLOW_EXPERIMENT_NAME

export MLFLOW_EXPERIMENT_NAME="fastText-0"

mlflow run ~/work/torch-fastText/ --entry-point=fasttext \
    --env-manager=local \
    -P remote_server_uri=$MLFLOW_TRACKING_URI \
    -P experiment_name=$MLFLOW_EXPERIMENT_NAME