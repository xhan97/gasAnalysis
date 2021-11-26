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

import pandas as pd
import numpy as np
import os
import time


def vwap(data):
    if data["Trade_Quantity"].sum() == 0:
        return np.nan
    else:
        return (data["Trade_Price"] * data["Trade_Quantity"]).sum() / data["Trade_Quantity"].sum()


def get_price(df):
    st = time.time()
    price_df = df.groupby(["Contract_Delivery_Date", pd.Grouper(freq='min')]).agg(Low_Price=('Trade_Price', 'min'),
                                                                                  High_Price=(
        'Trade_Price', 'max'),
        Open_Price=(
        'Trade_Price', 'first'),
        Close_Price=(
        "Trade_Price", "last"),
        Trade_Quantity=(
        "Trade_Quantity", 'sum'),
    )
    price_df["Vwap"] = df.groupby(
        ["Contract_Delivery_Date", pd.Grouper(freq='min')], group_keys=False).apply(vwap).tolist()
    price_df.dropna(how="any", inplace=True)
    et = time.time()
    print(st - et)
    return price_df


if __name__ == '__main__':
    archive_input_dir = "data\\raw\\Archive"
    selected_dir_list = ["2015", "2016",
                         "2017", "2018", "2019", "2020", "2021"]
    out_dir = "data\\interim\\Archive"
    os.makedirs(out_dir, exist_ok=True)
    for input_dir in selected_dir_list:
        subfiles = []
        for dirpath, subdirs, files in os.walk(os.path.join(archive_input_dir, input_dir)):
            for x in files:
                if x.endswith("eth.gz"):
                    subfiles.append(os.path.join(dirpath, x))
        dfs = []
        for file in subfiles:
            data = pd.read_csv(file, usecols=[0, 1, 6, 7, 9], parse_dates=[
                [0, 1]], header=None, sep=',', quotechar='"')
            data.rename(columns={'0_1': "Trade_Datetime", 6: "Contract_Delivery_Date",
                        7: "Trade_Quantity", 9: "Trade_Price"}, inplace=True)
            data.set_index("Trade_Datetime", inplace=True)
            dfs.append(data)
        concat_df = pd.concat(dfs)
        price_df = get_price(concat_df)
        price_df.to_csv(os.path.join(out_dir, input_dir+"_price.csv"),
                        float_format='%.3f', index=True)
