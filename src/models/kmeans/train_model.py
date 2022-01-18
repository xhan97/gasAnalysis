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

from tslearn.clustering import TimeSeriesKMeans
import os
from tslearn.utils import to_time_series_dataset


def dba_fit_predict_data(n_cluster, ts_dataset, save_model_path=None):
    seed = 13
    dba_km = TimeSeriesKMeans(n_clusters=n_cluster,
                              metric="dtw",
                              n_jobs=10,
                              max_iter_barycenter=10,
                              verbose=False,
                              n_init=2,
                              random_state=seed)
    y_pred = dba_km.fit_predict(ts_dataset)
    if save_model_path:
        file_name = "dba"+"_"+str(n_cluster)
        save_model_path = os.path.join(save_model_path, 'dba')
        os.makedirs(save_model_path, exist_ok=True)
        dba_km.to_pickle(os.path.join(save_model_path, file_name+'.pkl'))
    return dba_km, y_pred


def dba_fit_predict_vwap(n_cluster, data, save_model_path=None):
    period_vwap = to_time_series_dataset(data["Normal_Vwap"].values)
    km_model, y_pred = dba_fit_predict_data(
        n_cluster, period_vwap, save_model_path=save_model_path)
    return km_model, y_pred
