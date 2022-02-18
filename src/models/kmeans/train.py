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

from tslearn.utils import to_time_series_dataset


class Trainer:
    def __init__(self, model):
        """Initialize the trainer"""
        self.model = model

    def get_model(self):
        return self.model
    
    def train(self, dataset):
        #period_vwap = to_time_series_dataset(dataset["Normal_Vwap"].values)
        """Trains the model and logs the results"""
        self.model = self.model.fit(dataset)
        return self.get_model