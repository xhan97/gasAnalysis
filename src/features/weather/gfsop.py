# Copyright 2021 Administrator
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

import numpy as np
import pandas as pd

from weather import Weather


class Gfsop(Weather):
    def __init__(self, path) -> None:
        self.name = 'GFSOP'
        self.window_size = 15
        self.sub_delta = -6  # error set
        self.dst_hour = 0
        self.dst_minutes = 13
        self.norm_hour = 0
        self.norm_minutes = -47
        self.path = path

    def load_data(self, start_date):
        self.start_date = start_date
        self.data = pd.read_csv(
            self.path, sep=',', na_values=['-'], skiprows=[0])
        self.set_init_hour()
        return self

    def trans_dst(self):
        self.merge_df["Trans_INIT_Time"] = self.merge_df["INIT_Time"].apply(
            lambda x:  self.trans_dst_helper(x))
        self.merge_df['Trans_INIT_Time'] = (
            np.select(
                condlist=[self.merge_df['INIT_HOUR'] == "00:00:00"],
                choicelist=[self.merge_df['Trans_INIT_Time'] +
                            timedelta(hours=0, minutes=-2)],
                default=self.merge_df['Trans_INIT_Time']))
        return self

    def get_name(self):
        return self.name


if __name__ == '__main__':
    GFSOP_weather = Gfsop(
        path='data/raw/WeatherData/GFSOP_WDD_Forecasts_20100101_20210331.csv.gz')
    merge_df = GFSOP_weather.load_data(start_date='2015-01-01').merge_data(
    ).trans_dst().get_delta().get_merge_df(save="gfsop_weather_subclass.csv")
