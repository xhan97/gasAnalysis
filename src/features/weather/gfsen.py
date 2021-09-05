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

import numpy as np

from weather import Weather


class Gfsen(Weather):
    def __init__(self, path) -> None:
        self.name = 'GFSEN'
        self.window_size = 15
        self.sub_delta = -6
        self.dst_hour = 1
        self.dst_minutes = 28
        self.norm_hour = 0
        self.norm_minutes = 55
        self.path = path

    def set_init_hour(self):
        self.data['INIT_HOUR'] = (
            np.select(
                condlist=[self.data['INIT_HOUR'] == 0, self.data['INIT_HOUR'] ==
                          6, self.data['INIT_HOUR'] == 12, self.data['INIT_HOUR'] == 18],
                choicelist=["00:00:00", "06:00:00", "12:00:00", "18:01:00"],
                default="00:00:00"))
        return self

    def get_name(self):
        return self.name


if __name__ == '__main__':
    GFSEN_weather = Gfsen(
        path='data/raw/WeatherData/GFSEN_WDD_Forecasts_20100101_20210331.csv.gz')
    merge_df = GFSEN_weather.load_data(start_date='2015-01-01').merge_data(
    ).transform_dst().get_delta().get_merge_df(save="gfsen_weather_subclass.csv")
