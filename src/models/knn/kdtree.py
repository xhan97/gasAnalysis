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

import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from src.data.weather.ecmen import Ecmen
from src.data.weather.ecmop import Ecmop
from src.data.weather.gfsen import Gfsen
from src.data.weather.gfsop import Gfsop
from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import MinMaxScaler
import os


def load_data(data_path):
    data = pd.read_pickle(data_path, compression='gzip')
    return data


# def get_label(data: pd.DataFrame, km_model: TimeSeriesKMeans):
#     period_vwap = to_time_series_dataset(data["Normal_Vwap"].values)
#     labels = km_model.predict(period_vwap)
#     data["label"] = labels
#     return data


def knn_fit_predict(data: pd.DataFrame, n_neighbors: int, new_data: pd.DataFrame):
    features = new_data.columns
    features = [fe for fe in features if fe in data.columns]
    trans = MinMaxScaler()
    knc = KNeighborsClassifier(n_neighbors=n_neighbors)

    x_train = data[features]
    x_train = trans.fit_transform(x_train)
    knc.fit(x_train)
    x = new_data[features]
    x = trans.transform(x)
    kneig = knc.kneighbors(x, return_distance=True)
    neigbor_time = data.index[kneig[1][0]].tolist()
    dist = kneig[0][0].tolist()
    neibors = list(zip(neigbor_time, dist))
    return neibors


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
    new_data.set_index("Trans_INIT_Time", inplace=True)
    return new_data





if __name__ == '__main__':
    data = load_data(
        "E:/tulip/Misc/gasAnalysis/data/processd/predict/ecmen/06_00_13_40/ecmen_period_1_label.pkl.gz")
    km_model = TimeSeriesKMeans.from_pickle(
        "E:/tulip/Misc/gasAnalysis/models/k-means/ecmen/dba/dba_16.pkl")
    new_data = preprocess_new_data(
        'ecmen', "E:/tulip/Misc/gasAnalysis/data/raw/newdata/newecmen.csv")
    neibors = knn_fit_predict(data, 10, new_data)
    
    # save_data_path = 'data/processed/predict/ecmen/06_00_13_40'
    # os.makedirs(save_data_path, exist_ok=True)
    # data_basename = os.path.basename('E:/tulip/Misc/gasAnalysis/data/processed/period/ecmen/06_00_13_40/ecmen_period_1.pkl.gz')
    # data.to_pickle(os.path.join(save_data_path, data_basename[:-7]+"_label"+".pkl.gz"),compression='gzip')
    
    
