# Copyright 2022 Xin Han
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
import pandas as pd
from sklearn.neighbors import KDTree, KNeighborsClassifier
from src.data.weather.ecmen import Ecmen
from src.data.weather.ecmop import Ecmop
from src.data.weather.gfsen import Gfsen
from src.data.weather.gfsop import Gfsop
from tslearn.clustering import TimeSeriesKMeans
from tslearn.utils import to_time_series_dataset
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def load_data(data_path):
    data = pd.read_pickle(data_path, compression='gzip')
    data.set_index("Trans_INIT_Time",inplace=True)
    return data


def get_label(data: pd.DataFrame, km_model: TimeSeriesKMeans):
    period_vwap = to_time_series_dataset(data["Normal_Vwap"].values)
    labels = km_model.predict(period_vwap)
    data["label"] = labels
    return data


def knn_fit_predict(data: pd.DataFrame, n_neighbors: int, new_data: pd.DataFrame):
    features = new_data.columns
    features = [fe for fe in features if fe in data.columns]
    pipe = Pipeline([('scaler', StandardScaler()), ('knc', KNeighborsClassifier(n_neighbors=n_neighbors))])
    x_train = data[features]
    y_train = data['label']
    pipe.fit(x_train, y_train)
    x = new_data[features]
    kneig = pipe.kneighbors(x, return_distance=True)
    return kneig


def preprocess_new_data(weather_name, weather_path,):
    if weather_name == 'ecmen':
        weather = Ecmen(path=weather_path)
    elif weather_name == 'ecmop':
        weather = Ecmop(path=weather_path)
    elif weather_name == 'gfsop':
        weather = Gfsop(path=weather_path)
    elif weather_name == 'gfsen':
        weather = Gfsen(path=weather_path)
    else:
        raise NotImplementedError
    new_data = weather.load_data().merge_data.transform_dst.get_delta.get_df()
    new_data["Month"] = new_data["Trans_INIT_Time"].dt.month
    new_data.set_index("Trans_INIT_Time",inplace=True)
    return new_data


if __name__ == '__main__':
    data = load_data(
        "E:/tulip/Misc/gasAnalysis/data/processed/period/ecmen/06_00_13_40/ecmen_period_1.pkl.gz")
    km_model = TimeSeriesKMeans.from_pickle("E:/tulip/Misc/gasAnalysis/models/k-means/ecmen/dba/dba_16.pkl")
    data = get_label(data, km_model)
    
    new_data = preprocess_new_data('ecmen', "E:/tulip/Misc/gasAnalysis/data/raw/newdata/newecmen.csv")
    
    # weather_feature = data[['VALUE_hdd', '10Y_NORMAL_hdd', 'Sum_Value_hdd', 'Delta_Full_hdd',
    #                         'Delta_Sub_hdd', 'VALUE_cdd', '10Y_NORMAL_cdd', 'Sum_Value_cdd',
    #                         'Delta_Full_cdd', 'Delta_Sub_cdd', 'Month']]
    # print(data)
    knn_fit_predict(data,10,new_data)
