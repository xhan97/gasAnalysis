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

import math
import os
from datetime import timedelta

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset


period_data = pd.read_csv("data/processed/period/ecmen/6_16/ecmenperiod1.csv.gz", compression='gzip',)


def dba_fit_predict_data(n_cluster, period_data, data_name, save_model_path=None):
    seed = 13
    dba_km = TimeSeriesKMeans(n_clusters=n_cluster,
                              metric="dtw",
                              n_jobs=8,
                              max_iter_barycenter=10,
                              verbose=False,
                              n_init=2,
                              random_state=seed)
    y_pred = dba_km.fit_predict(period_data)
    if save_model_path:
        file_name = "dba_"+data_name+"_"+str(n_cluster)
        dba_km.to_pickle(os.path.join(save_model_path, file_name+".pkl"))
    return y_pred

