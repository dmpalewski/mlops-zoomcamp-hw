import pandas as pd

from datetime import datetime, timedelta
from pathlib import Path
import pickle
from typing import Union

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import flow, task, get_run_logger
from prefect.deployments import DeploymentSpec
from prefect.orion.schemas.schedules import CronSchedule
from prefect.flow_runners import SubprocessFlowRunner


DATA_DIR = Path(__file__).parents[1] / "data"
MODELS_DIR = Path(__file__).parents[1] / "models"


@task
def get_paths(date: datetime):
    logger = get_run_logger()
    val_date = date.replace(day=1) - timedelta(days=1)
    train_date = val_date.replace(day=1) - timedelta(days=1)
    val_path = DATA_DIR / f"fhv_tripdata_{val_date.strftime('%Y-%m')}.parquet"
    train_path = DATA_DIR / f"fhv_tripdata_{train_date.strftime('%Y-%m')}.parquet"
    logger.info(f"Training data: {train_path}")
    logger.info(f"Validation data: {val_path}")
    return train_path, val_path

@task
def read_data(path):
    df = pd.read_parquet(path)
    return df

@task
def prepare_features(df, categorical, train=True):
    logger = get_run_logger()
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    mean_duration = df.duration.mean()
    if train:
        logger.info(f"The mean duration of training is {mean_duration}")
    else:
        logger.info(f"The mean duration of validation is {mean_duration}")
    
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    return df

@task
def train_model(df, categorical):
    logger = get_run_logger()
    train_dicts = df[categorical].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(train_dicts) 
    y_train = df.duration.values

    logger.info(f"The shape of X_train is {X_train.shape}")
    logger.info(f"The DictVectorizer has {len(dv.feature_names_)} features")

    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_train)
    mse = mean_squared_error(y_train, y_pred, squared=False)
    logger.info(f"The MSE of training is: {mse}")
    return lr, dv

@task
def run_model(df, categorical, dv, lr):
    logger = get_run_logger()
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts) 
    y_pred = lr.predict(X_val)
    y_val = df.duration.values

    mse = mean_squared_error(y_val, y_pred, squared=False)
    logger.info(f"The MSE of validation is: {mse}")
    return

@flow
def main(date: Union[None, str]="2021-08-15"):
    logger = get_run_logger()
    date_obj = datetime.fromisoformat(date) if date else datetime.today()
    categorical = ['PUlocationID', 'DOlocationID']
    train_path, val_path = get_paths(date_obj).result()

    df_train = read_data(train_path)
    df_train_processed = prepare_features(df_train, categorical)

    df_val = read_data(val_path)
    df_val_processed = prepare_features(df_val, categorical, False)

    # train the model
    lr, dv = train_model(df_train_processed, categorical).result()
    trained_model_path = MODELS_DIR / f"model-{date_obj.strftime('%Y-%m-%d')}.bin"
    trained_vectorizer_path = MODELS_DIR / f"dv-{date_obj.strftime('%Y-%m-%d')}.b"
    with open(trained_model_path, "wb") as fout:
        pickle.dump(lr, fout)
    logger.info(f"Model saved to {trained_model_path}")
    with open(trained_vectorizer_path, "wb") as fout:
        pickle.dump(dv, fout)
    logger.info(f"DictVectorizer saved to {trained_vectorizer_path}")
    run_model(df_val_processed, categorical, dv, lr)


DeploymentSpec(
    name="cron-schedule-deployment",
    flow=main,
    schedule=CronSchedule(cron="0 9 15 * *", timezone="CET"),
    flow_runner=SubprocessFlowRunner(),
    tags=["cron"]
)
