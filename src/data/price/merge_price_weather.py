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


import os

import numpy as np
import pandas as pd

from datetime import datetime, timedelta


def get_weather_public_time(df, weather_col_name, weather_path):
    df[weather_col_name] = np.nan
    weather_data = pd.read_csv(weather_path)
    time_sep = weather_data["Trans_INIT_Time"].drop_duplicates(
    ).reset_index(drop=True).values
    cutoff_pair = list(zip(time_sep, time_sep[1:]))
    for pair in cutoff_pair:
        df[weather_col_name].loc[pair[0]:pair[1]] = pair[0]
    df[weather_col_name] = pd.to_datetime(
        df[weather_col_name])
    return df


def get_dst_flag(df):
    df["dst_flag"] = 0
    dst_pairs = [(datetime.strptime("2015-3-8", '%Y-%m-%d'),
                  datetime.strptime("2015-11-1", '%Y-%m-%d')),
                 (datetime.strptime("2016-3-13", '%Y-%m-%d'),
                  datetime.strptime("2016-11-6", '%Y-%m-%d')),
                 (datetime.strptime("2017-3-12", '%Y-%m-%d'),
                  datetime.strptime("2017-11-5", '%Y-%m-%d')),
                 (datetime.strptime("2018-3-11", '%Y-%m-%d'),
                  datetime.strptime("2018-11-4", '%Y-%m-%d')),
                 (datetime.strptime("2019-3-10", '%Y-%m-%d'),
                  datetime.strptime("2019-11-3", '%Y-%m-%d')),
                 (datetime.strptime("2020-3-8", '%Y-%m-%d'),
                  datetime.strptime("2020-11-1", '%Y-%m-%d')),
                 (datetime.strptime("2021-3-14", '%Y-%m-%d'),
                  datetime.strptime("2021-11-7", '%Y-%m-%d'))]
    for item in dst_pairs:
        df["dst_flag"].loc[item[0]:item[1]] = 1
    return df


def get_period(time_series, start_time, end_time):
    st_time = datetime.strptime(start_time, '%H:%M').time()
    ed_time = datetime.strptime(end_time, '%H:%M').time()
    time_point = [st_time, ed_time]
    public_time = time_series.dt.time.unique()
    for item in public_time:
        if (item >= st_time) & (item <= ed_time):
            time_point.append(item)
    time_point.sort()
    time_point_pairs = list(zip(time_point, time_point[1:]))
    return time_point_pairs


def normal_vmap(df):
    peirod_grouped = df[['Contract_Delivery_Date', "Vwap"]].groupby(
        ['Contract_Delivery_Date', pd.Grouper(freq='D')])
    return peirod_grouped.transform(lambda x: x / x.values[0])


if __name__ == '__main__':

    output_dir = "data/processed/period"
    weather_name = "gfsop"
    weather_path = "data/processed/WeatherData/gfsop_weather_subclass.csv"
    cut_off_df_path = "data/interim/Archive/cut_off/cut_off_price.csv"
    start_time = "6:00"
    end_time = "16:00"

    cut_off_df = pd.read_csv(cut_off_df_path, parse_dates=[
        "Trade_Datetime"], index_col="Trade_Datetime")
    weather_col_name = weather_name+"_pulicate_time"
    cut_off_df = get_weather_public_time(
        cut_off_df, weather_col_name, weather_path)
    cut_off_df = get_dst_flag(cut_off_df)
    cut_off_df.index = cut_off_df.index + pd.DateOffset(hours=-1)

    cut_off_df.loc[cut_off_df.dst_flag == 1, weather_col_name] = cut_off_df[cut_off_df["dst_flag"]
                                                                            == 1][weather_col_name].apply(lambda x: x + timedelta(hours=-1))
    cut_off_df.dropna(how='any', inplace=True)
    periods = get_period(
        cut_off_df[weather_col_name], start_time=start_time, end_time=end_time)
    for i, pair in enumerate(periods, start=0):
        period_data = cut_off_df[["Contract_Delivery_Date", "Vwap", weather_col_name, "dst_flag"]].between_time(
            pair[0], pair[1], include_start=False)
        period_data["Normal_Vwap"] = normal_vmap(period_data)
        save_path = os.path.join(output_dir, weather_name, "_".join(
            [t.split(":")[0] for t in [start_time, end_time]]))
        os.makedirs(save_path, exist_ok=True)
        period_data.to_csv(os.path.join(save_path, weather_name+"period"+str(i+1)+".csv.gz"),
                           float_format='%.3f',
                           index=True,
                           compression='gzip')
