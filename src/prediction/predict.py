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

from tslearn.clustering import TimeSeriesKMeans
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from matplotlib.gridspec import GridSpec
from dotenv import find_dotenv, load_dotenv
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import click
import logging
import math
import os
import sys
from pathlib import Path

sys.path.append('src')
from data.weather.gfsop import Gfsop
from data.weather.gfsen import Gfsen
from data.weather.ecmop import Ecmop
from data.weather.ecmen import Ecmen

def lighten_color(color, amount=0.5):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """
    import colorsys

    import matplotlib.colors as mc
    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    return colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])


def load_data(data_path):
    data = pd.read_pickle(data_path, compression='gzip')
    return data


def merge_gnsd(trained_data, gnsd_path):
    trained_data.reset_index(inplace=True)
    gnsd_data = pd.read_csv(gnsd_path, parse_dates=['date'])
    trained_data['weather_public_date'] = trained_data['weather_public_time'].dt.date
    gnsd_data['weather_public_date'] = gnsd_data['date'].dt.date
    trained_data = trained_data.merge(gnsd_data, on='weather_public_date', how='inner').drop(
        ['weather_public_date', 'date', 'weather_public_time'], axis=1)
    trained_data.set_index("Trans_INIT_Time", inplace=True)
    return trained_data


def get_neibors(data: pd.DataFrame, n_neighbors: int, new_data: pd.DataFrame):
    features = new_data.columns
    features = [fe for fe in features if fe in data.columns]
    trans = MinMaxScaler()
    knc = KNeighborsClassifier(n_neighbors=n_neighbors)

    x_train = data[features]
    x_train = trans.fit_transform(x_train)
    y_train = data['label']
    knc.fit(x_train, y_train)

    x = new_data[features]
    x = trans.transform(x)
    kneig = knc.kneighbors(x, return_distance=True)
    neigbor_time = data.index[kneig[1][0]].tolist()
    dist = kneig[0][0].tolist()
    neibors_vwap = data.loc[neigbor_time]['Normal_Vwap'].apply(lambda x: x[-1])
    neibors = dict(zip(neigbor_time, dist))
    return neibors, neibors_vwap


def get_neibors_values(data: pd.DataFrame, neibors: dict):

    neibors_times = neibors.keys.tolist()


def normalize_time(series):
    series = pd.to_datetime(series, format="%H:%M:%S").to_series()
    series += pd.to_timedelta(series.lt(series.shift()).cumsum(), unit="D")
    return series.values


def show_time_series(period_df: pd.DataFrame, label: int, km_model, kn_item: dict, ax_fig):
    max_index = []
    k = kn_item.keys()
    i = 0
    dark_color = 'black'
    normal_color = [0.7, 0.8, 0.8]

    label_period = period_df[period_df["label"]
                             == label][["Normal_Vwap", "Trade_time"]]
    index_list = label_period.index.to_list()
    neibor_index = {index_list.index(
        item): kn_item[item] for item in k if item in index_list}
    for Normal_Vwap, Trade_time in label_period.values:
        Trade_time = normalize_time(Trade_time)
        if len(Trade_time) > len(max_index):
            max_index = Trade_time
        if i in neibor_index.keys():
            # sns.lineplot(x=Trade_time, y=Normal_Vwap, color=lighten_color(
            # dark_color, neibor_index[i]), linewidth = 2, ax=ax_fig)
            sns.lineplot(x=Trade_time, y=Normal_Vwap,
                         color=dark_color, linewidth=1, ax=ax_fig)
        # sns.lineplot(x=Trade_time, y=Normal_Vwap,
        #              color=normal_color, ax=ax_fig)
        i += 1

    center_df = pd.DataFrame(
        {"Time": max_index, "normal_vwap": km_model.cluster_centers_[label].ravel()})
    sns.lineplot(x="Time", y="normal_vwap", data=center_df,
                 color="red", ax=ax_fig)
    ax_fig.set_title("Cluster "+str(label))
    ax_fig.xaxis.set_major_formatter(
        matplotlib.dates.DateFormatter("%H:%M")
    )


def save_clustering_fig(km_model, merge_data, kn_item: dict, neibors_vwap, save_path):
    fig = plt.figure(figsize=(40, 30))
    n_clusters = km_model.n_clusters
    plot_count = math.ceil(math.sqrt(n_clusters))
    gs = GridSpec(plot_count, plot_count+1, figure=fig)
    row_i = 0
    column_j = 0
    for i, label in enumerate(reversed(range(n_clusters))):
        ax = fig.add_subplot(gs[row_i, column_j])
        show_time_series(period_df=merge_data, km_model=km_model, kn_item=kn_item,
                         label=label, ax_fig=ax)
        column_j += 1
        if column_j % plot_count == 0:
            row_i += 1
            column_j = 0
    ax = fig.add_subplot(gs[1:3, -1])
    sns.violinplot(data=neibors_vwap, ax=ax)
    ax.set_title('Distribution  of  the last vwap of black line')
    if save_path:
        os.makedirs(save_path,  exist_ok=True)
        plt.savefig(os.path.join(save_path, "predict.pdf"),
                    bbox_inches="tight", orientation='landscape')


def preprocess_new_weather_data(weather_name, weather_path):
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
    new_weather_data = weather.load_data().merge_data.transform_dst.get_delta.get_df()
    new_weather_data["Month"] = new_weather_data["Trans_INIT_Time"].dt.month
    return new_weather_data


def preprocess_new_gnsd_data(gnsd_path):
    gnsd_data = pd.read_csv(gnsd_path, parse_dates=["date"])
    return gnsd_data


def preprocess_new_data(weather_name, weather_path, gnsd_path):
    if os.path.exists(weather_path) and os.path.exists(gnsd_path):
        new_weather_data = preprocess_new_weather_data(
            weather_name, weather_path)
        new_gnsd_data = preprocess_new_gnsd_data(gnsd_path)
        new_weather_data['public_date'] = new_weather_data['Trans_INIT_Time'].dt.date
        new_gnsd_data['public_date'] = new_gnsd_data['date'].dt.date
        new_data = new_weather_data.merge(new_gnsd_data, on='public_date', how='inner').drop(
            ['public_date', 'date'], axis=1)
        new_data.set_index("Trans_INIT_Time", inplace=True)
        return new_data
    elif os.path.exists(weather_path):
        new_weather_data = preprocess_new_weather_data(
            weather_name, weather_path)
        new_weather_data.set_index("Trans_INIT_Time", inplace=True)
        return new_weather_data
    elif os.path.exists(gnsd_path):
        new_gnsd_data = preprocess_new_gnsd_data(gnsd_path)
        new_gnsd_data.set_index("date", inplace=True)
        return new_gnsd_data
    else:
        raise ValueError("one of weather_path and gnsd_path must be exist")


@click.command()
@click.option('--trained_data_path', default="data/processed/predict/ecmen/06_00_13_40/ecmen_period_1_label.pkl.gz", type=click.Path(exists=True))
@click.option('--model_path', default='models/k-means/ecmen/dba/dba_16.pkl', type=click.Path(exists=True))
@click.option('--new_weather_data_path', default='data/raw/newdata/newecmen.csv', type=click.Path())
@click.option('--new_weather_data_name', default='ecmen')
@click.option('--gnsd_historical_data_path', default='data/processed/gnsdData/gnsd.csv', type=click.Path())
@click.option('--new_gnsd_data_path', default='data/raw/newdata/newgnsd.csv', type=click.Path())
@click.option('--k', default=10)
@click.option('--save_figure_path', default='reports/figures/kmeansCluster/ecmen')
def main(trained_data_path, model_path, new_weather_data_path, new_weather_data_name, gnsd_historical_data_path, new_gnsd_data_path, k, save_figure_path):
    """ Runs data processing scripts to turn raw data from (../raw) into
        cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__name__)
    logger.info('predicting new data')
    os.makedirs(save_figure_path, exist_ok=True)

    data = load_data(trained_data_path)
    data = merge_gnsd(data, gnsd_historical_data_path)

    km_model = TimeSeriesKMeans.from_pickle(model_path)
    new_data = preprocess_new_data(
        new_weather_data_name, new_weather_data_path, new_gnsd_data_path)
    neibors, neibors_vwap = get_neibors(data, k, new_data)

    save_clustering_fig(km_model, data, neibors,
                        neibors_vwap, save_figure_path)
    logger.info('predicted figure is saved in ' + save_figure_path + ' !')


if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())
    main()
