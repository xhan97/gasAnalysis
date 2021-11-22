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


def cal_delta(x):
    if x[-1, -1]:
        da = x[-1, 0] - (x[0, 0])
    else:
        da = x[-1, 0] - (x[0, 1] + x[-1, 2])
    return da


class Weather(object):
    def __init__(self, path, window_size, sub_delta, dst_hour, dst_minutes, norm_hour, norm_minutes) -> None:
        self._path = path
        self._window_size = window_size
        self._sub_delta = sub_delta
        self._df = None
        self._dst_hour = dst_hour
        self._dst_minutes = dst_minutes
        self._norm_hour = norm_hour
        self._norm_minutes = norm_minutes

    def load_data(self, start_date):
        self._data = pd.read_csv(self._path, sep=',', na_values=['-'])
        self._set_init_hour()
        self._start_date = start_date
        return self

    def _set_init_hour(self):
        self._data['INIT_HOUR'] = (
            np.select(
                condlist=[self._data['INIT_HOUR'] == 0, self._data['INIT_HOUR'] ==
                          6, self._data['INIT_HOUR'] == 12, self._data['INIT_HOUR'] == 18],
                choicelist=["00:00:00", "06:00:00", "12:00:00", "18:00:00"],
                default="00:00:00"))
        return self

    @property
    def get_data(self):
        return self._data

    @property
    def get_public_time(self):
        return self._public_time

    @property
    def merge_data(self):
        window_grouper = self._data["VALUE"].index // self._window_size
        self._df = self._data[["INIT_DATE", "INIT_HOUR", "VERIF_DATE", "PARAM", "VALUE",
                               "10Y_NORMAL"]].groupby(window_grouper).head(1)
        self._df["Sum_Sub_Value"] = self._data[["VALUE"]].groupby(window_grouper).apply(
            lambda x: sum(x["VALUE"].values[self._sub_delta:])).values
        self._df["Sum_Sub_tail"] = self._data[["VALUE"]].groupby(window_grouper).apply(
            lambda x: sum(x["VALUE"].values[(self._sub_delta+1):])).values
        self._df["Sum_Value"] = self._data[["VALUE"]
                                           ].groupby(window_grouper).sum().values
        self._df["10Y_NORMAL_LAST"] = self._data[[
            "10Y_NORMAL"]].groupby(window_grouper).tail(1).values
        self._df["SUB_First"] = self._data[["VALUE"]].groupby(window_grouper).apply(
            lambda x: x["VALUE"].values[self._sub_delta]).values
        self._df["INIT_Time"] = self._df["INIT_DATE"] + \
            " " + self._df["INIT_HOUR"]
        self._df["INIT_Time"] = pd.to_datetime(
            self._df["INIT_Time"], format="%Y-%m-%d %H:%M:%S")
        self._df = self._df[(
            self._df['INIT_Time'] >= self._start_date)]
        self._df["VALUE"] = self._df["VALUE"].astype("float64")
        return self

    def _trans_dst_helper(self, dt):
        if (((dt >= datetime.strptime("2015-3-8", '%Y-%m-%d')) & (dt < datetime.strptime("2015-11-1", '%Y-%m-%d'))) |
            ((dt >= datetime.strptime("2016-3-13", '%Y-%m-%d')) & (dt < datetime.strptime("2016-11-6", '%Y-%m-%d'))) |
            ((dt >= datetime.strptime("2017-3-12", '%Y-%m-%d')) & (dt < datetime.strptime("2017-11-5", '%Y-%m-%d'))) |
            ((dt >= datetime.strptime("2018-3-11", '%Y-%m-%d')) & (dt < datetime.strptime("2018-11-4", '%Y-%m-%d'))) |
            ((dt >= datetime.strptime("2019-3-10", '%Y-%m-%d')) & (dt < datetime.strptime("2019-11-3", '%Y-%m-%d'))) |
            ((dt >= datetime.strptime("2020-3-8", '%Y-%m-%d')) & (dt < datetime.strptime("2020-11-1", '%Y-%m-%d'))) |
                ((dt >= datetime.strptime("2021-3-14", '%Y-%m-%d')) & (dt < datetime.strptime("2021-11-7", '%Y-%m-%d')))):
            dt = dt + timedelta(hours=self._dst_hour,
                                minutes=self._dst_minutes)
        else:
            dt = dt + timedelta(hours=self._norm_hour,
                                minutes=self._norm_minutes)
        return dt

    @property
    def transform_dst(self):
        self._df["Trans_INIT_Time"] = self._df["INIT_Time"].apply(
            lambda x:  self._trans_dst_helper(x))
        return self

    def __get_time_flag(self, x):
        return x.iloc[-1] == x.iloc[0]

    def calcute_delta(self, data):
        data["Tail_Value"] = data["Sum_Value"] - data["VALUE"]
        data["VERIF_DATE"] = pd.to_datetime(data["VERIF_DATE"])
        data["Time_Flag"] = data["VERIF_DATE"].view(np.int64).rolling(
            window=2).apply(self.__get_time_flag).astype(bool).values
        data["Delta_Full"] = data[["Sum_Value", "Tail_Value", "10Y_NORMAL_LAST", "Time_Flag"]].rolling(
            2, method="table", min_periods=0).apply(cal_delta, raw=True, engine="numba")["Sum_Value"]
        data["Delta_Sub"] = data[["Sum_Sub_Value", "Sum_Sub_tail", "10Y_NORMAL_LAST", "Time_Flag"]].rolling(
            2, method="table", min_periods=0).apply(cal_delta, raw=True, engine="numba")["Sum_Sub_Value"]
        return data[['Trans_INIT_Time', 'VALUE', '10Y_NORMAL', 'Sum_Value', 'Delta_Full', "Delta_Sub"]]

    @property
    def get_delta(self):
        hdd_data = self._df[self._df["PARAM"] == "HDD"].reset_index()
        cdd_data = self._df[self._df["PARAM"] == "CDD"].reset_index()
        cdd_data_delta = self.calcute_delta(cdd_data)
        hdd_data_delta = self.calcute_delta(hdd_data)
        self._df = pd.merge(hdd_data_delta, cdd_data_delta, how="inner",
                            on='Trans_INIT_Time', suffixes=('_hdd', '_cdd'))
        self._df.sort_values(by=['Trans_INIT_Time'], inplace=True)
        return self

    def get_df(self, save=False) -> pd.DataFrame:
        if save:
            self._df.to_csv(save, header=True, index=False,
                            float_format='%.2f')
        return self._df


if __name__ == '__main__':

    pass
