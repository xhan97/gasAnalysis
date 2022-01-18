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

from email.policy import default
import logging
from pathlib import Path

import click
import pandas as pd
from dotenv import find_dotenv, load_dotenv

from train_model import dba_fit_predict_vwap
from visualize import show_clustering


def load_data(data_path):
    data = pd.read_pickle(data_path, compression='gzip')
    return data


@click.command()
@click.argument('data_path', default='data/processed/period/ecmen/06_00_13_40/ecmen_period_1.pkl.gz', type=click.Path(exists=True))
@click.argument('num_clusters', default=16, type=click.INT)
@click.argument('save_model_path', default='models/k-means/ecmen')
@click.argument('save_figure_path', default='reports/figures/kmeansCluster/ecmen')
def main(data_path, num_clusters, save_model_path, save_figure_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('traing k-means model')

    data = load_data(data_path)
    dba_model, y_hat = dba_fit_predict_vwap(
        data=data, n_cluster=num_clusters, save_model_path=save_model_path)
    data["label"] = y_hat

    logger.info('Visualizing clusters of k-means')
    show_clustering(km_model=dba_model, n_clusters=num_clusters,
                    merge_data=data, save_path=save_figure_path)


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
