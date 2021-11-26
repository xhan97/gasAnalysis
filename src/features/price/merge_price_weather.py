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

weather_input_dir = 'data/processed/WeatherData/'
output_dir = 'data/processed'

cut_off_data = pd.read_csv('data/interim/Archive/cut_off/cut_off_price.csv', parse_dates=[
    "Trade_Datetime"], index_col="Trade_Datetime")

weather_paths = []
for dirpath, subdirs, files in os.walk(weather_input_dir):
    for x in files:
        if x.endswith(".csv"):
            weather_paths.append((x[:5], os.path.join(dirpath, x)))

for weather_pt in weather_paths:
    cl_name = weather_pt[0]+"_pulicate_time"
    weather_data = pd.read_csv(weather_pt[1])
    cut_off_data[cl_name] = np.nan
    time_sep = weather_data["Trans_INIT_Time"].drop_duplicates(
    ).reset_index(drop=True).values
    cutoff_pair = list(zip(time_sep, time_sep[1:]))
    for pair in cutoff_pair:
        cut_off_data[cl_name].loc[pair[0]:pair[1]] = pair[0]

weather_col_names = [item[0] + "_pulicate_time" for item in weather_paths]
cut_off_data.dropna(how='all', subset=weather_col_names, inplace=True)
cut_off_data.to_csv(os.path.join(
    output_dir, "cut_off_price_with_weather_time.csv"), float_format='%.3f', index=True)
