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


from weather import Weather


class Ecmop(Weather):
    def __init__(self, path) -> None:
        self.name = 'ECMOP'
        self.window_size = 9
        self.sub_delta = -5  # error set
        self.dst_hour = 1
        self.dst_minutes = 55
        self.norm_hour = 0
        self.norm_minutes = 55
        self.path = path

    def get_name(self):
        return self.name


if __name__ == '__main__':
    ECMOP_weather = Ecmop(
        path='data/raw/WeatherData/ECMOP_WDD_Forecasts_20100101_20210331.csv.gz')
    merge_df = ECMOP_weather.load_data(start_date='2015-01-01').merge_data(
    ).transform_dst().get_delta().get_merge_df(save="ecmop_weather_subclass.csv")
