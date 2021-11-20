#!/usr/bin/env python
# coding: utf-8


import glob
import pandas as pd
import numpy as np
import os


dirlist = ["2018", "2019", "2020", "2021"]

shpfiles = []
for inputdir in dirlist:
    print(inputdir)
    subfiles = []
    for dirpath, subdirs, files in os.walk(inputdir+"/"):
        for x in files:
            if x.endswith(".gz"):
                subfiles.append(os.path.join(dirpath, x))
    shpfiles.append(subfiles)


def vwap(data):
    if data.volume.sum() == 0:
        return np.nan
    else:
        return (data.price * data.volume).sum() / data.volume.sum()


def get_price(sub_data):
    price_df = sub_data.groupby([pd.Grouper(freq='min')]).min()
    price_df.rename(columns={"price": "low_price"}, inplace=True)
    price_df["volume"] = sub_data["volume"].groupby(
        [pd.Grouper(freq='min')]).sum().tolist()
    price_df["high_price"] = sub_data["price"].groupby(
        [pd.Grouper(freq='min')]).max().tolist()
    price_df["open_price"] = sub_data["price"].groupby(
        [pd.Grouper(freq='min')]).first().tolist()
    price_df["close_price"] = sub_data["price"].groupby(
        [pd.Grouper(freq='min')]).last().tolist()
    price_df["vmap"] = sub_data.groupby(
        [pd.Grouper(freq='min')], group_keys=False).apply(vwap).tolist()
    price_df.dropna(how="any", inplace=True)
    return price_df[["volume", "low_price", "high_price", "open_price", "close_price", "vmap"]]


for fileitem in shpfiles:
    for file in fileitem:
        data = pd.read_csv(file, usecols=[0, 1, 6, 7, 9], parse_dates=[
                           [0, 1]], header=None, sep=',', quotechar='"')
        data.rename(columns={'0_1': 'datetime', 6: 'day',
                    7: 'volume', 9: 'price'}, inplace=True)
        data.set_index("datetime", inplace=True)
        day_m = pd.unique(data['day']).tolist()
        for item in day_m:
            subdf = data[data["day"] == item]
            price_df = get_price(subdf)
            save_path = "result"+"/" + file[:10]
            os.makedirs(save_path, exist_ok=True)
            price_df.to_csv(save_path+"/"+str(item)+"price.csv",
                            header=True, index=True, float_format='%.3f')


CONCAT_DIR = "result/globel/"
os.makedirs(CONCAT_DIR, exist_ok=True)

for item in dirlist:
    # Use glob module to return all csv files under root directory. Create DF from this.
    files = pd.DataFrame([file for file in glob.glob(
        "result/"+item+"/*/*/*")], columns=["fullpath"])

    #    fullpath
    # 0  root\dir1\data_20170101_k.csv
    # 1  root\dir1\data_20170102_k.csv
    # 2  root\dir2\data_20170101_k.csv
    # 3  root\dir2\data_20170102_k.csv

    # # Split the full path into directory and filename
    files_split = files['fullpath'].str.rsplit(
        "/", 1, expand=True).rename(columns={0: 'path', 1: 'filename'})

    #    path       filename
    # 0  root\dir1  data_20170101_k.csv
    # 1  root\dir1  data_20170102_k.csv
    # 2  root\dir2  data_20170101_k.csv
    # 3  root\dir2  data_20170102_k.csv

    # Join these into one DataFrame
    files = files.join(files_split)
    print(files)

    #    fullpath                       path        filename
    # 0  root\dir1\data_20170101_k.csv  root\dir1   data_20170101_k.csv
    # 1  root\dir1\data_20170102_k.csv  root\dir1   data_20170102_k.csv
    # 2  root\dir2\data_20170101_k.csv  root\dir2   data_20170101_k.csv
    # 3  root\dir2\data_20170102_k.csv  root\dir2   data_20170102_k.csv

#     # Iterate over unique filenames; read CSVs, concat DFs, save file
    for f in files['filename'].unique():
        paths = files[files['filename'] == f]['fullpath']
        dfs = [pd.read_csv(path) for path in paths]
        concat_df = pd.concat(dfs) 
        concat_df["CC_date"] = f[:4]
        save_path = os.path.join(CONCAT_DIR, item)
        os.makedirs(save_path, exist_ok=True)
        concat_df.to_csv(os.path.join(save_path, f), index=False,
                         float_format='%.3f')  

CONCAT_DIR = "result/globel/"
for item in dirlist:
    filelist = [file for file in glob.glob("result/globel/"+item+"/*")]
    # Get list of dataframes from CSV file paths
    dfs = [pd.read_csv(path) for path in filelist]
    concat_df = pd.concat(dfs)  # Concat dataframes into one
    # concat_df.sort_index(inplace=True)
    concat_df.sort_values(by=['datetime'], inplace=True)
    save_path = CONCAT_DIR
    os.makedirs(save_path, exist_ok=True)
    concat_df.to_csv(save_path + item + "_price.csv", index=False,
                     float_format='%.3f')  # Save dataframe
