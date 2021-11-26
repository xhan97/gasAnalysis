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
        self._name = 'GFSEN'
        super().__init__(path=path,
                         window_size=15,
                         sub_delta=-6,
                         dst_hour=1,
                         dst_minutes=28,
                         norm_hour=0,
                         norm_minutes=55)

    def _init_hour(self):
        self._data['INIT_HOUR'] = (
            np.select(
                condlist=[self._data['INIT_HOUR'] == 0, self._data['INIT_HOUR'] ==
                          6, self._data['INIT_HOUR'] == 12, self._data['INIT_HOUR'] == 18],
                choicelist=["00:00:00", "06:00:00", "12:00:00", "18:01:00"],
                default="00:00:00"))
        return self

    @property
    def get_name(self):
        return self._name


if __name__ == '__main__':
    import os
    outdir = "data\processed\WeatherData"
    GFSEN_weather = Gfsen(
        path='data/raw/WeatherData/GFSEN_WDD_Forecasts_20100101_20210331.csv.gz')
    merge_df = GFSEN_weather. \
                load_data(start_date='2015-01-01'). \
                merge_data. \
                transform_dst. \
                get_delta. \
                get_df(save=os.path.join(outdir,"gfsen_weather_subclass.csv"))
