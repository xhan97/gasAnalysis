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
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


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


def get_weather_public_time(cutoff_df, weather_data):
    weather_col_name = "weather_public_time"
    cutoff_df[weather_col_name] = np.nan
    time_sep = weather_data["Trans_INIT_Time"].drop_duplicates(
    ).reset_index(drop=True).values
    cutoff_pair = list(zip(time_sep, time_sep[1:]))
    for pair in cutoff_pair:
        cutoff_df[weather_col_name].loc[pair[0]:pair[1]] = pair[0]
    cutoff_df[weather_col_name] = pd.to_datetime(
        cutoff_df[weather_col_name])
    cutoff_df.dropna(how='any', inplace=True)
    return cutoff_df


def merge_period_cutoff_weather(period_data, weather_data):
    period_data["Trade_time"] = period_data.index.time
    daily_period_data = period_data.groupby(['Contract_Delivery_Date', pd.Grouper(freq='D')]).agg(Normal_Vwap=('Normal_Vwap', list),
                                                                                                  Trade_time=(
        'Trade_time', list),
        weather_public_time=("weather_public_time", 'first'))
    weather_data["Trans_INIT_Time"] = pd.to_datetime(
        weather_data["Trans_INIT_Time"])
    merge_data = pd.merge(daily_period_data, weather_data, how="left", left_on="weather_public_time", right_on='Trans_INIT_Time',
                          suffixes=('_hdd', '_cdd'))
    merge_data["Month"] = merge_data["Trans_INIT_Time"].dt.month
    return merge_data


def make_period_cutoff(weather_name, weather_path,  cutoff_path, st_time, ed_time, using_period, out_dir):
    cutoff_df = pd.read_csv(cutoff_path, parse_dates=[
        "Trade_Datetime"], index_col="Trade_Datetime")
    weather_data = pd.read_csv(weather_path)
    cutoff_with_weather_public_time = get_weather_public_time(
        cutoff_df, weather_data)

    periods = get_period(
        cutoff_with_weather_public_time["weather_public_time"], start_time=st_time, end_time=ed_time)
    try:
        selected_period_pair = periods[using_period-1]
    except:
        print("f{using_period} is not available!")
        raise ValueError
    period_data = cutoff_with_weather_public_time[["Contract_Delivery_Date", "Vwap", "weather_public_time"]].between_time(
        selected_period_pair[0], selected_period_pair[1], include_start=False)
    period_data["Normal_Vwap"] = normal_vmap(period_data)

    merged_period = merge_period_cutoff_weather(
        period_data=period_data, weather_data=weather_data)

    save_path = os.path.join(out_dir, weather_name, "_".join(
        [t.strftime('%H_%M') for t in selected_period_pair]))
    os.makedirs(save_path, exist_ok=True)
    merged_period.to_pickle(os.path.join(
        save_path, weather_name+"_period_"+str(using_period)+".pkl.gz"), compression='gzip')
