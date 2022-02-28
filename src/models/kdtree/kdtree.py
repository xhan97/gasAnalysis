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
import numpy as np

from sklearn.neighbors import KDTree


def load_data(data_path):
    data = pd.read_pickle(data_path, compression='gzip')
    data = data.head(10)
    weather_feature = data[['VALUE_hdd', '10Y_NORMAL_hdd', 'Sum_Value_hdd', 'Delta_Full_hdd',
                            'Delta_Sub_hdd', 'VALUE_cdd', '10Y_NORMAL_cdd', 'Sum_Value_cdd',
                            'Delta_Full_cdd', 'Delta_Sub_cdd', 'Month']]
    return weather_feature


def build_kdtree(X: np.array, metric, leaf_size):
    tree = KDTree(X, leaf_size=leaf_size, metric=metric)
    return tree


def kd_query(X: np.array, kd_model, k: int):
    dist, ind = kd_model.query(X, k=k)
    return dist, ind


if __name__ == '__main__':
    data = load_data(
        "data/processed/period/ecmen/06_00_13_40/ecmen_period_1.pkl.gz")
    print(data)
