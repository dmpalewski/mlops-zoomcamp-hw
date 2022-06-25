#!/usr/bin/env python
# coding: utf-8

import argparse
import pickle
import pandas as pd


with open('model.bin', 'rb') as f_in:
    dv, lr = pickle.load(f_in)


categorical = ['PUlocationID', 'DOlocationID']

def read_data(filename):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.dropOff_datetime - df.pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df

def prepare_output(input, predictions, year, month):
    df = pd.DataFrame()
    df['ride_id'] = f'{year:04d}/{month:02d}_' + input.index.astype('str')
    df['predictions'] = predictions
    return df


def main(args):
    year = args.year
    month = args.month


    df = read_data(f'https://nyc-tlc.s3.amazonaws.com/trip+data/fhv_tripdata_{year:04d}-{month:02d}.parquet')


    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = lr.predict(X_val)


    pred_avg = sum(y_pred) / len(y_pred)
    print(f'Average prediction for {year:04d}-{month:02d} is {pred_avg:.2f}')


    df_result = prepare_output(input=df, predictions=y_pred, year=year, month=month)
    df_result.to_parquet(
        "output/h4q2.parquet",
        engine='pyarrow',
        compression=None,
        index=False
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, help='Year of the dataset to be used.')
    parser.add_argument('--month', type=int, help='Month of the dataset to be used.')
    args = parser.parse_args()
    main(args)
