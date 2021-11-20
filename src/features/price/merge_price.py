# Copyright 2021 Xin Han
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

#!/usr/bin/env python
# coding: utf-8


import glob
import os

import pandas as pd

dirlist = ["2015", "2016", "2017", "2018", "2019", "2020", "2021"]

shpfiles = []
for inputdir in dirlist:
    print(inputdir)
    subfiles = []
    for dirpath, subdirs, files in os.walk(inputdir+"\\"):
        for x in files:
            if x.endswith(".gz"):
                subfiles.append(os.path.join(dirpath, x))
    shpfiles.append(subfiles)


def vwap(data):
    return (data.price * data.volume).sum() / data.volume.sum()


def get_price(sub_data):
    price_df = sub_data.groupby("minete").min()
    price_df.rename(columns={"price": "low_price"}, inplace=True)
    price_df["volume"] = sub_data.groupby("minete").sum()["volume"].tolist()
    price_df["high_price"] = sub_data.groupby("minete").max()["price"].tolist()
    price_df["open_price"] = sub_data.groupby("minete").first()[
        "price"].tolist()
    price_df["close_price"] = sub_data.groupby("minete").last()[
        "price"].tolist()
    price_df["vmap"] = sub_data.groupby("minete", group_keys=False).apply(vwap)
    price_df.reset_index(inplace=True)
    price_df["date"] = price_df["date"].astype(str)
    price_df["datetime"] = price_df["date"] + " " + price_df["minete"]+":00"
    features = ["datetime", "volume", "low_price", "high_price", "open_price", "close_price", "vmap"]
    return price_df[features]


for fileitem in shpfiles:
    for file in fileitem:
        data = pd.read_csv(file, usecols=[0, 1, 6, 7, 9], names=[
                           "date", "time", "day", "volume", "price"], header=None, sep=',', quotechar='"')
        day_m = pd.unique(data['day']).tolist()
        for item in day_m:
            subdf = data[data["day"] == item]
            subdf["minete"] = subdf["time"].astype(str).str[0:5]
            subdf.drop(columns=["time"], inplace=True)
            price_df = get_price(subdf)
            savepath = "result"+"//" + file[:10]
            os.makedirs(savepath, exist_ok=True)
            price_df.to_csv(savepath+"//"+str(item)+"price.csv",
                            header=True, index=False, float_format='%.3f')

CONCAT_DIR = "result//globel//"
os.makedirs(CONCAT_DIR, exist_ok=True)

for item in dirlist:
    files = pd.DataFrame([file for file in glob.glob(
        "result\\"+item+"\\*\\*\\*")], columns=["fullpath"])
    files_split = files['fullpath'].str.rsplit(
        "\\", 1, expand=True).rename(columns={0: 'path', 1: 'filename'})

    files = files.join(files_split)
    print(files)
    for f in files['filename'].unique():
        paths = files[files['filename'] == f]['fullpath']
        dfs = [pd.read_csv(path) for path in paths]
        concat_df = pd.concat(dfs)  # Concat dataframes into one
        concat_df["CC_date"] = f[:4]
        savepath = CONCAT_DIR + item + "//"
        os.makedirs(savepath, exist_ok=True)
        concat_df.to_csv(savepath+f, index=False,
                         float_format='%.3f')  # Save dataframe

CONCAT_DIR = "result//globel//"
for item in dirlist:
    filelist = [file for file in glob.glob("result\\globel\\"+item+"\\*")]
    # Get list of dataframes from CSV file paths
    dfs = [pd.read_csv(path) for path in filelist]
    concat_df = pd.concat(dfs)  # Concat dataframes into one
    savepath = CONCAT_DIR
    os.makedirs(savepath, exist_ok=True)
    concat_df.to_csv(savepath + item + "_price.csv", index=False,
                     float_format='%.3f')  # Save dataframe


if __name__ == '__main__':
    pass