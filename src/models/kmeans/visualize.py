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

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.gridspec import GridSpec
from scipy.stats import norm


def show_time_series(period_df, label, km_model, ax_fig):
    max_index = []
    for Normal_Vwap, Trade_time in period_df[period_df["label"] == label][["Normal_Vwap", "Trade_time", ]].values:
        color = [0.7, 0.8, 0.8]
        Trade_time = normalize_time(Trade_time)
        if len(Trade_time) > len(max_index):
            max_index = Trade_time
        sns.lineplot(x=Trade_time, y=Normal_Vwap, color=color, ax=ax_fig)
    center_df = pd.DataFrame(
        {"Time": max_index, "normal_vwap": km_model.cluster_centers_[label].ravel()})
    sns.lineplot(x="Time", y="normal_vwap", data=center_df,
                 color="red", ax=ax_fig)
    ax_fig.set_title("Cluster "+str(label))
    ax_fig.xaxis.set_major_formatter(
        matplotlib.dates.DateFormatter("%H:%M")
    )


def normalize_time(series):
    series = pd.to_datetime(series, format="%H:%M:%S").to_series()
    series += pd.to_timedelta(series.lt(series.shift()).cumsum(), unit="D")
    return series.values


def get_ci_df(merge_data):
    stat_feature = ['Sum_Value_hdd', 'Delta_Full_hdd', 'Delta_Sub_hdd',
                    'Sum_Value_cdd', 'Delta_Full_cdd', 'Delta_Sub_cdd']
    stats = merge_data.groupby(['label'])[stat_feature].agg([
        'mean', 'count', 'std'])
    ci = 0.95
    ci_data = []
    for label in stats.index:
        for fe in stat_feature:
            mean, count, std = stats.loc[label, fe]
            z_hat = norm.ppf(ci+(1-ci)/2)*std/math.sqrt(count)
            ci_left, ci_right = mean-z_hat, mean+z_hat
            ci_data.append({'label': label, 'feature': fe,
                            'ci_left': ci_left, 'ci_right': ci_right})
    ci_df = pd.DataFrame(ci_data)
    return ci_df


def show_ci_box(ci_df, label, ax_fig):
    query_res = []
    ci_df_index_label = ci_df.set_index(['label'])
    for feature, ci_left, ci_right in ci_df_index_label.loc[label].values:
        ci_dict = {feature: "[{:.2f}, {:.2f}]".format(ci_left, ci_right)}
        query_res.append("\n".join("{}: {}".format(*i)
                                   for i in ci_dict.items()))
    query_i = 0
    ax_fig.axis('off')
    ax_fig.set_title("Confidence Interval")
    for item in np.arange(0.05, 0.95, 0.15).tolist():
        ax_fig.text(0.1, item, query_res[query_i],
                    size=14)
        query_i += 1


def show_confidence_interval(df, ax):
    ci_d = df.round(2)[["ci_left", "ci_right"]]
    se_hat_x_list = list(ci_d.itertuples(index=False, name=None))
    x_hat_list = [(x[0]+x[1])/2 for x in se_hat_x_list]
    for i in range(len(x_hat_list)):
        ax.errorbar(x_hat_list[i], np.arange(len(x_hat_list))[
                    i], lolims=True, xerr=se_hat_x_list[i][1]-x_hat_list[i], yerr=0.0, linestyle='', c='black', elinewidth=3, mew=3)


def show_month_distribution(merge_data, label, ax_fig):
    label_df = merge_data[merge_data['label'] == label]
    sns.countplot(x="Month", data=label_df, ax=ax_fig)


def show_clustering(km_model, n_clusters, merge_data, save_path):
    fig = plt.figure(figsize=(50, 80))
    gs = GridSpec(n_clusters, 9, figure=fig)

    ci_df = get_ci_df(merge_data)
    for i, label in enumerate(reversed(range(16))):
        ax = fig.add_subplot(gs[i, 0])
        show_time_series(period_df=merge_data, km_model=km_model,
                         label=label, ax_fig=ax)
        ax2 = fig.add_subplot(gs[i, 1])
        show_ci_box(ci_df, label, ax_fig=ax2)
        ax3 = fig.add_subplot(gs[i, -1])
        show_month_distribution(merge_data, label=label, ax_fig=ax3)
    i = 2
    for feature, df in ci_df.groupby("feature"):
        ax = fig.add_subplot(gs[:, i])
        show_confidence_interval(df, ax)
        ax.set_title(feature)
        i += 1
    plt.savefig(save_path+".pdf",
                bbox_inches="tight", orientation='landscape')
