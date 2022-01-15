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

from datetime import timedelta
from dateutil.relativedelta import relativedelta
import pandas as pd
import numpy as np
import os


def vwap(data):
    if data["Trade_Quantity"].sum() == 0:
        return np.nan
    else:
        return (data["Trade_Price"] * data["Trade_Quantity"]).sum() / data["Trade_Quantity"].sum()


def get_price(df):
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
    price_df.reset_index(level = ['Contract_Delivery_Date'],inplace=True)
    return price_df


def get_merge_price(archive_dirs, select_period):
    all_price = []
    for period in select_period:
        eth_paths = []
        for dirpath, _, files in os.walk(os.path.join(archive_dirs, period)):
            for x in files:
                if x.endswith("eth.gz"):
                    eth_paths.append(os.path.join(dirpath, x))
        for file_path in eth_paths:
            data = pd.read_csv(file_path, usecols=[0, 1, 6, 7, 9], parse_dates=[
                [0, 1]], header=None, sep=',', quotechar='"')
            data.rename(columns={'0_1': "Trade_Datetime", 6: "Contract_Delivery_Date",
                        7: "Trade_Quantity", 9: "Trade_Price"}, inplace=True)
            data.set_index("Trade_Datetime", inplace=True)
            all_price.append(data)
    merge_price = pd.concat(all_price)
    return merge_price


def get_cutoff(price_df):
    cutoff_df = price_df.groupby(pd.Grouper(freq='M')).apply(lambda x: (x.index.max(
    ) + timedelta(days=-2)).replace(hour=13, minute=29, second=0)).to_frame("cutoff_date")
    cutoff_df["cutoff_date"] = cutoff_df["cutoff_date"].astype(str)
    cutoff_df["cutoff_date_last"] = cutoff_df["cutoff_date"].shift(periods=1)
    cutoff_df["cc_date"] = [(item + relativedelta(months=+1)
                             ).strftime('%Y%m')[-4:] for item in cutoff_df.index]
    cutoff_df.dropna(how='any', inplace=True)

    dfs = []
    for item in cutoff_df[["cutoff_date_last", "cutoff_date", "cc_date"]].values:
        qdata = price_df.loc[item[0]:item[1]].query(
            "Contract_Delivery_Date==" + item[2])
        dfs.append(qdata)
    concat_df = pd.concat(dfs)
    return concat_df


def make_price(achive_dirs, output_dir, st_year):
    os.makedirs(output_dir, exist_ok=True)
    selected_dir_list = [item for item in os.listdir(
        achive_dirs) if int(item) >= int(st_year)]
    merged_price_df = get_merge_price(achive_dirs, selected_dir_list)
    price_df = get_price(merged_price_df)
    print(price_df.head())
    cut_off_df = get_cutoff(price_df)
    cut_off_df.to_csv(os.path.join(output_dir, "cut_off_price.csv"),
                      float_format='%.3f', index=True)


if __name__ == '__main__':
    archive_input_dir = "data\\raw\\Archive"
    out_dir = "data\\interim\\Archive\\price"
    make_price(archive_input_dir,out_dir,'2021')