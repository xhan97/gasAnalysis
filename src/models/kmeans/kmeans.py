#!/usr/bin/env python
# coding: utf-8

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

dirlist = ["2015", "2016", "2017", "2018", "2019", "2020", "2021"]
shpfiles = []
for inputdir in dirlist:
    for dirpath, subdirs, files in os.walk(inputdir+"/"):
        for x in files:
            if x.endswith(".csv"):
                shpfiles.append(os.path.join(dirpath, x))


weather_data = pd.read_csv(
    "../../WeatherData/GFSEN_WDD_Forecasts.csv", parse_dates=["Trans_INIT_Time"])
time_sep = weather_data["Trans_INIT_Time"].drop_duplicates(
).reset_index(drop=True).values
cutoff_pair = list(zip(time_sep, time_sep[1:]))

first_period, secound_period = cutoff_pair[::4], cutoff_pair[1::4]
theird_period, fourth_perid = cutoff_pair[2::4], cutoff_pair[3::4]


period_dict = {"first_period": first_period,
               'secound_period': secound_period, 'theird_period': theird_period}
period_date = {"first_period": "00:28:00-06:28:00",
               'secound_period': "06:28:00-12:28:00", 'theird_period': "12:28:00"}

weather_data.set_index("Trans_INIT_Time", inplace=True)


def nomalize(sequence, method="l1"):
    if method == "True":
        res = [item/sequence[0] for item in sequence]
    elif method == "l2":
        res = [j/i for i, j in zip(sequence[:-1], sequence[1:])]
    return res


def get_value_set(file_list, cutoff_pair, time_flag, use_nomalize=False):
    li = []
    for file_path in file_list:
        data = pd.read_csv(file_path, parse_dates=[
            "datetime"], index_col="datetime")
        for item in cutoff_pair:
            df = None
            df = data.loc[item[0]:item[1]]
            if item[0].astype(str)[11:19] == time_flag:
                df["time_flag"] = 1
            else:
                df["time_flag"] = 0
            if df is not None:
                li.append(df)
    concat_df = pd.concat(li, axis=0)
    concat_df.dropna(how="any", inplace=True)
    res = concat_df.groupby(['CC_date', pd.Grouper(freq='D')]).agg(list)[
        'vmap'].to_list()
    if use_nomalize:
        res = [nomalize(item, method=use_nomalize)
               for item in res if len(item) > 2]
    return concat_df, to_time_series_dataset(res)


def normalize_time(series):
    series = pd.to_datetime(series, format="%H:%M:%S").to_series()

    series += pd.to_timedelta(series.lt(series.shift()).cumsum(), unit="D")
    # print(series)
    return series.values


def fit_data(n_cluster, period_data, period_df, file_name, method="dba", use_nomalize=False):
    if method == "dba":
        seed = 13
        dba_km = TimeSeriesKMeans(n_clusters=n_cluster,
                                  metric="dtw",
                                  n_jobs=8,
                                  max_iter_barycenter=10,
                                  verbose=False,
                                  n_init=2,
                                  random_state=seed)
        y_pred = dba_km.fit_predict(period_data)
        sava_path = "dba_"+file_name+"_"+str(n_cluster)+"_"+str(use_nomalize)
        # print(sava_path)
        dba_km.to_pickle(sava_path+".pkl")
        show_clustering(dba_km=dba_km, n_clusters=n_cluster,
                        y_pred=y_pred, period_data=period_data, period_df=period_df, file_name=sava_path)


def get_cluster_time_set(dba_km, y_pred, period_df):

    time_dict = {}

    for label in set(y_pred):
        i = 0
        time_set = []
        for _, d in period_df.groupby(['CC_date', pd.Grouper(freq='D')]):
            if len(d) > 2:
                if(y_pred[i] == label):
                    time_set.append(d.index)
                i += 1
        time_dict.update({label: time_set})
    return time_dict


def show_confidence_interval(ci_df, query_str, filename):
    fig, ax = plt.subplots(1, 1, figsize=(6, 8))
    fig.suptitle(" ".join(filename))
    ci_sub_query = ci_df.round(2).query(query_str)[["left_ci", "right_ci"]]
    SE_hat_x_list = list(ci_sub_query.itertuples(index=False, name=None))
    x_hat_list = [(x[0]+x[1])/2 for x in SE_hat_x_list]
    for i in range(len(x_hat_list)):
        ax.errorbar(x_hat_list[i], np.arange(len(x_hat_list))[
                    i], lolims=True, xerr=SE_hat_x_list[i][1]-x_hat_list[i], yerr=0.0, linestyle='', c='black')
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig("_".join(filename)+".png", dpi=200,
                bbox_inches="tight", orientation='landscape')


def show_distribution(concat_df, file_name):
    fig, axs = plt.subplots(2, 3, figsize=(20, 20))
    fig.suptitle(" ".join(file_name))
    row_i = 0
    column_j = 0
    for PARAM in ["CDD", "HDD"]:
        sub_df = concat_df[concat_df["PARAM"] == PARAM]
        for item in ["Sum_Value", "Delta_Sub", "Delta_Full"]:
            sns.distplot(sub_df[item], rug=True, rug_kws={"color": "g"},
                         kde_kws={"color": "k", "lw": 3, "label": "KDE"},
                         hist_kws={"histtype": "step", "linewidth": 3,
                                   "alpha": 1, "color": "g"}, ax=axs[row_i, column_j])
            axs[row_i, column_j].set_title(PARAM)
            column_j += 1
            if column_j % 3 == 0:
                row_i += 1
                column_j = 0
    fig.tight_layout()
    fig.subplots_adjust(top=0.95)
    plt.savefig("_".join(file_name)+".png", dpi=200,
                bbox_inches="tight", orientation='landscape')


def get_max_index(period_df):
    len_max_index = 0
    max_index = None
    for _, d in period_df.groupby(['CC_date', pd.Grouper(freq='D')]):
        if len(d) > 2:
            d["normal_vmap"] = nomalize(d["vmap"].values, method="True")
            data_index = d.index
            if d["time_flag"][0] == 1:
                data_index = data_index + timedelta(hours=-1, minutes=0)
            d["Time"] = normalize_time(data_index.time)
            if len(d["Time"]) > len_max_index:
                len_max_index = len(d["Time"])
                max_index = d["Time"]
    return len_max_index, max_index


def show_time_series(period_df, axs_fig, label):
    i = 0
    for _, d in period_df.groupby(['CC_date', pd.Grouper(freq='D')]):
        if len(d) > 2:
            if(y_pred[i] == label):
                d["normal_vmap"] = nomalize(
                    d["vmap"].values, method="True")
                data_index = d.index
                color = [0.7, 0.8, 0.8]
                if d["time_flag"][0] == 1:
                    data_index = data_index + \
                        timedelta(hours=-1, minutes=0)
                    color = [1, 0.96, 0.56]
                d["Time"] = normalize_time(data_index.time)

                sns.lineplot(x="Time", y="normal_vmap", data=d,
                             color=color, ax=axs_fig)
            i += 1


def show_ci_box(ci_df, ax_fig, label):
    query_res = []
    for param in ["CDD", "HDD"]:
        for fe in ["Sum_Value", "Delta_Sub", "Delta_Full"]:
            ci_query = ci_df.round(2).query(f"label == {label} & Param=='{param}' & fe=='{fe}'")[
                ["Param", "fe", "left_ci", "right_ci"]].iloc[0].to_dict()
            ci_query.update(
                {"ci": [ci_query["left_ci"], ci_query["right_ci"]]})
            for key in ['left_ci', 'right_ci']:
                ci_query.pop(key)
            query_res.append("\n".join("{}: {}".format(*i)
                                       for i in ci_query.items()))
    query_i = 0
    for item in np.arange(0.05, 0.95, 0.15).tolist():
        ax_fig.text(1.05, item, query_res[query_i],
                    size=10,
                    bbox=dict(
            edgecolor='lightgreen', facecolor='none', pad=3, linewidth=1),
            ha='left', va='center', transform=ax_fig.transAxes)
        query_i += 1


def show_cluster_center(max_index, dba_km, label, axs_fig):
    center_df = pd.DataFrame(
        {"index": max_index.values, "data": dba_km.cluster_centers_[label].ravel()})
    sns.lineplot(x="index", y="data", data=center_df,
                 color="red", ax=axs_fig)


def show_clustering(dba_km, n_clusters, y_pred, ci_df, period_df, file_name):
    plot_count = math.ceil(math.sqrt(n_clusters))
    fig, axs = plt.subplots(plot_count, plot_count, figsize=(20, 20))
    row_i = 0
    column_j = 0

    len_max_index, max_index = get_max_index(period_df)
    for label in set(y_pred):
        show_time_series(period_df=period_df, label=label,
                         axs_fig=axs[row_i, column_j])

        assert len_max_index == len(dba_km.cluster_centers_[
                                    label].ravel()), "time max index length error!"
        show_cluster_center(max_index, dba_km, label=label,
                            axs_fig=axs[row_i, column_j])
        show_ci_box(ci_df, label=label, ax_fig=axs[row_i, column_j])

        axs[row_i, column_j].set_title(
            "Cluster "+str(row_i*plot_count+column_j))
        axs[row_i, column_j].xaxis.set_major_formatter(
            matplotlib.dates.DateFormatter("%H:%M")
        )
        column_j += 1
        if column_j % plot_count == 0:
            row_i += 1
            column_j = 0
    fig.tight_layout(pad=1.0)
    plt.savefig(file_name+".png", dpi=200,
                bbox_inches="tight", orientation='landscape')


def get_confidence_interbval(sample, ci=0.95):
    x_hat = np.mean(sample)
    n = len(sample)
    z_hat = norm.ppf(ci+(1-ci)/2)*np.std(sample)/np.sqrt(n)
    return x_hat-z_hat, x_hat+z_hat


for period in list(period_dict.keys())[0:1]:
    for normal in ["True", ]:
        if period == "first_period":
            time_flag = "01:28:00"
            model_path = "../K-means/first_period/models/dba_first_period_16_True.pkl"
        elif period == "secound_period":
            time_flag = "07:28:00"
            model_path = "../K-means/secound_period/models/dba_secound_period_16_True.pkl"
        elif period == "theird_period":
            time_flag = "13:28:00"
            model_path = "../K-means/theird_period/models/dba_theird_period_16_True.pkl"
        period_df, period_data = get_value_set(
            shpfiles, period_dict[period], time_flag=time_flag, use_nomalize=normal)
        model = TimeSeriesKMeans.from_pickle(model_path)
        y_pred = model.predict(period_data)

        res = get_cluster_time_set(model, y_pred, period_df)

        ci_data = []
        for label in res.keys():
            li = []
            for item in res[label]:
                li.append(weather_data[weather_data.index.isin(item)])
            concat_df = pd.concat(li, axis=0)
            concat_df = concat_df.loc[concat_df.index.strftime(
                "%H:%M:%S") <= time_flag]

            show_distribution(concat_df, file_name=[period, str(label)])

            for PARAM in ["CDD", "HDD"]:
                for fe in ["Sum_Value", "Delta_Sub", "Delta_Full"]:
                    sub_df = concat_df[concat_df["PARAM"] == PARAM]

                    left_ci, right_ci = get_confidence_interbval(
                        sub_df[fe].values)
                    ci_data.append((label, PARAM, fe, left_ci, right_ci))

        ci_df = pd.DataFrame(
            ci_data, columns=["label", "Param", "fe", "left_ci", "right_ci"])

        for PARAM in ["CDD", "HDD"]:
            for fe in ["Sum_Value", "Delta_Sub", "Delta_Full"]:
                query_str = f"Param=='{PARAM}' & fe=='{fe}'"
                show_confidence_interval(ci_df, query_str, [period, PARAM, fe])

        show_clustering(dba_km=model,
                        y_pred=y_pred,
                        n_clusters=16,
                        file_name=model_path[:-4],
                        period_df=period_df,
                        ci_df=ci_df)
