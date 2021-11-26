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

"""
Front month contract cutoff 在每个月倒数第三个交易日下午13：29分。
比如某个月最后一个交易时间是27日，那就找到25日的13：29作为cutoff。
对于某月m, 首先找到在m-1月的cutoff时间c1和在m月的cutoff时间c2。
在c1-c2期间发生的交割期为m+1的交易，为Front month trade。
比如对于2021年1月，从2020年12月29日13：30到2021年1月25日13：29的交易中，选取交割时间为2102的所有数据为 Front month trade。 

根据此规则整合所有价格数据，提取Front month trade。
"""


import os
from datetime import timedelta

import pandas as pd
from dateutil.relativedelta import relativedelta

price_input_dir = "data\\interim\\Archive"
out_dir = "data\\processed\\Archive"

price_files_path = []
for dirpath, subdirs, files in os.walk(price_input_dir):
    for x in files:
        if x.endswith("_price.csv"):
            price_files_path.append(os.path.join(dirpath, x))

li = []
for file_path in price_files_path:
    df = pd.read_csv(file_path,
                     parse_dates=["Trade_Datetime"], index_col="Trade_Datetime")
    li.append(df)
data = pd.concat(li, axis=0)


cutoff_df = data.groupby(pd.Grouper(freq='M')).apply(lambda x: (x.index.max(
) + timedelta(days=-2)).replace(hour=13, minute=29, second=0)).to_frame("cutoff_date")
cutoff_df["cutoff_date"] = cutoff_df["cutoff_date"].astype(str)
cutoff_df["cutoff_date_last"] = cutoff_df["cutoff_date"].shift(periods=1)
cutoff_df["cc_date"] = [(item + relativedelta(months=+1)
                         ).strftime('%Y%m')[-4:] for item in cutoff_df.index]
cutoff_df.dropna(how='any', inplace=True)


dfs = []
for item in cutoff_df[["cutoff_date_last", "cutoff_date", "cc_date"]].values:
    qdata = data.loc[item[0]:item[1]].query("Contract_Delivery_Date==" + item[2])
    dfs.append(qdata)
concat_df = pd.concat(dfs)

concat_df.to_csv(os.path.join(out_dir, "cut_off_price.csv"),
                 float_format='%.3f', index=True)