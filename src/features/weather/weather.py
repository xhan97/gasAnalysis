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

from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def get_delta(x):
    if x[-1, -1]:
        da = x[-1, 0] - (x[0, 0])
    else:
        da = x[-1, 0] - (x[0, 1] + x[-1, 2])
    return da


class Weather(object):
    def __init__(self, path, window_size, sub_delta, dst_hour, dst_minutes, norm_hour, norm_minutes) -> None:
        self.path = path
        self.window_size = window_size
        self.sub_delta = sub_delta
        self.merge_df = None
        self.dst_hour = dst_hour
        self.dst_minutes = dst_minutes
        self.norm_hour = norm_hour
        self.norm_minutes = norm_minutes

    def load_data(self, start_date):
        self.data = pd.read_csv(self.path, sep=',', na_values=['-'])
        self.set_init_hour()
        self.start_date = start_date
        return self

    def set_init_hour(self):
        self.data['INIT_HOUR'] = (
            np.select(
                condlist=[self.data['INIT_HOUR'] == 0, self.data['INIT_HOUR'] ==
                          6, self.data['INIT_HOUR'] == 12, self.data['INIT_HOUR'] == 18],
                choicelist=["00:00:00", "06:00:00", "12:00:00", "18:00:00"],
                default="00:00:00"))
        return self

    def merge_data(self):
        self.merge_df = self.data[["INIT_DATE", "INIT_HOUR", "VERIF_DATE", "PARAM", "VALUE",
                                   "10Y_NORMAL"]].groupby(self.data["VALUE"].index // self.window_size).head(1)
        self.merge_df["Sum_Sub_Value"] = self.data[["VALUE"]].groupby(
            self.data["VALUE"].index // self.window_size).apply(lambda x: sum(x["VALUE"].values[self.sub_delta:])).values
        self.merge_df["Sum_Sub_tail"] = self.data[["VALUE"]].groupby(
            self.data["VALUE"].index // self.window_size).apply(lambda x: sum(x["VALUE"].values[(self.sub_delta+1):])).values
        self.merge_df["Sum_Value"] = self.data[["VALUE"]].groupby(
            self.data["VALUE"].index // self.window_size).sum().values
        self.merge_df["10Y_NORMAL_LAST"] = self.data[["10Y_NORMAL"]].groupby(
            self.data["10Y_NORMAL"].index // self.window_size).tail(1).values
        self.merge_df["SUB_First"] = self.data[["VALUE"]].groupby(
            self.data["VALUE"].index // self.window_size).apply(lambda x: x["VALUE"].values[self.sub_delta]).values

        self.merge_df["INIT_Time"] = self.merge_df["INIT_DATE"] + \
            " " + self.merge_df["INIT_HOUR"]
        self.merge_df["INIT_Time"] = pd.to_datetime(
            self.merge_df["INIT_Time"], format="%Y-%m-%d %H:%M:%S")
        self.merge_df = self.merge_df[(
            self.merge_df['INIT_Time'] >= self.start_date)]
        self.merge_df["VALUE"] = self.merge_df["VALUE"].astype("float64")
        return self

    def trans_dst_helper(self, dt):
        if (((dt >= datetime.strptime("2015-3-8", '%Y-%m-%d')) & (dt < datetime.strptime("2015-11-1", '%Y-%m-%d'))) |
            ((dt >= datetime.strptime("2016-3-13", '%Y-%m-%d')) & (dt < datetime.strptime("2016-11-6", '%Y-%m-%d'))) |
            ((dt >= datetime.strptime("2017-3-12", '%Y-%m-%d')) & (dt < datetime.strptime("2017-11-5", '%Y-%m-%d'))) |
            ((dt >= datetime.strptime("2018-3-11", '%Y-%m-%d')) & (dt < datetime.strptime("2018-11-4", '%Y-%m-%d'))) |
            ((dt >= datetime.strptime("2019-3-10", '%Y-%m-%d')) & (dt < datetime.strptime("2019-11-3", '%Y-%m-%d'))) |
            ((dt >= datetime.strptime("2020-3-8", '%Y-%m-%d')) & (dt < datetime.strptime("2020-11-1", '%Y-%m-%d'))) |
                ((dt >= datetime.strptime("2021-3-14", '%Y-%m-%d')) & (dt < datetime.strptime("2021-11-7", '%Y-%m-%d')))):
            dt = dt + timedelta(hours=self.dst_hour, minutes=self.dst_minutes)
        else:
            dt = dt + timedelta(hours=self.norm_hour,
                                minutes=self.norm_minutes)
        return dt

    def transform_dst(self):
        self.merge_df["Trans_INIT_Time"] = self.merge_df["INIT_Time"].apply(
            lambda x:  self.trans_dst_helper(x))
        return self

    def get_time_flag(self, x):
        return x.iloc[-1] == x.iloc[0]

    def calcute_delta(self, data, col_name):
        sub_df = data[data["PARAM"]
                      == col_name].reset_index()
        sub_df["Tail_Value"] = sub_df["Sum_Value"] - sub_df["VALUE"]
        sub_df["VERIF_DATE"] = pd.to_datetime(sub_df["VERIF_DATE"])
        sub_df["Time_Flag"] = sub_df["VERIF_DATE"].view(np.int64).rolling(
            window=2).apply(self.get_time_flag).astype(bool).values
        sub_df["Delta_Full"] = sub_df[["Sum_Value", "Tail_Value", "10Y_NORMAL_LAST", "Time_Flag"]].rolling(
            2, method="table", min_periods=0).apply(get_delta, raw=True, engine="numba")["Sum_Value"]
        sub_df["Delta_Sub"] = sub_df[["Sum_Sub_Value", "Sum_Sub_tail", "10Y_NORMAL_LAST", "Time_Flag"]].rolling(
            2, method="table", min_periods=0).apply(get_delta, raw=True, engine="numba")["Sum_Sub_Value"]
        return sub_df[['Trans_INIT_Time', 'VALUE', '10Y_NORMAL', 'Sum_Value', 'Delta_Full', "Delta_Sub"]]

    def get_delta(self):
        hdd_data = self.calcute_delta(self.merge_df, "HDD")
        cdd_data = self.calcute_delta(self.merge_df, "CDD")
        self.merge_df = pd.merge(hdd_data, cdd_data, how="inner",
                                 on='Trans_INIT_Time', suffixes=('_hdd', '_cdd'))
        self.merge_df.sort_values(by=['Trans_INIT_Time'], inplace=True)

        return self

    def get_merge_df(self, save=False) -> pd.DataFrame:
        if save:
            self.merge_df.to_csv(save, header=True, index=False,
                                 float_format='%.2f')
        return self.merge_df


if __name__ == '__main__':

    pass
