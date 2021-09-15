#!/usr/bin/env python
# coding: utf-8

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dtreeviz.trees import *
from hypergbm import make_experiment
from scipy.stats import randint, uniform
from sklearn import tree
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, auc,
                             confusion_matrix, mean_squared_error)
from sklearn.model_selection import (GridSearchCV, KFold, RandomizedSearchCV,
                                     cross_val_score, train_test_split)
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight

import xgboost as xgb
from xgboost import plot_tree

data = pd.read_csv("weather_ECMEN_secound_period_cluster.csv",
                   parse_dates=['Trans_INIT_Time'])

def display_scores(scores):
    print("Scores: {0}\nMean: {1:.3f}\nStd: {2:.3f}".format(
        scores, np.mean(scores), np.std(scores)))

def report_best_scores(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results['rank_test_score'] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print("Mean validation score: {0:.3f} (std: {1:.3f})".format(
                  results['mean_test_score'][candidate],
                  results['std_test_score'][candidate]))
            print("Parameters: {0}".format(results['params'][candidate]))
            print("")

period_data = pd.read_csv(
    'secound_period_ECMEN_data.csv', parse_dates=['datetime'])

cdd_data = data[data["PARAM"] == "CDD"][[
    "Trans_INIT_Time", "Sum_Value", "Delta_Full", "Delta_Sub"]]
hdd_data = data[data["PARAM"] == "HDD"][[
    "Trans_INIT_Time", "Sum_Value", "Delta_Full", "Delta_Sub"]]
merge_data = pd.merge(hdd_data, cdd_data, how="inner",
                      on='Trans_INIT_Time', suffixes=('_hdd', '_cdd'))


merge_data['just_date'] = merge_data['Trans_INIT_Time'].dt.date
period_data['datetime'] = period_data['datetime'].dt.date
merge_df = pd.merge(merge_data, period_data, how="inner",
                    left_on='just_date', right_on='datetime')
merge_df.drop(columns=['just_date', 'datetime', 'time_flag'], inplace=True)

merge_df["Month"] = merge_df["Trans_INIT_Time"].dt.month
merge_df.drop(columns=['Trans_INIT_Time', 'CC_date',	'volume',
              'low_price',	'high_price',	'open_price',	'close_price'], inplace=True)

merge_df.to_csv("ecmen_secound_period_rate.csv",
                index=False, float_format='%.3f')

merge_df.label.value_counts()

bins = [0, 0.9, 1.1, np.infty]
merge_df["label"] = pd.cut(merge_df["vmap"], bins, labels=[
                           'plunge', 'normal', 'surge'])


sns.distplot(merge_df['vmap'], rug=True, rug_kws={"color": "g"},
             kde_kws={"color": "k", "lw": 3, "label": "KDE"},
             hist_kws={"histtype": "step", "linewidth": 3,
                       "alpha": 1, "color": "g"})

X = merge_df[["Sum_Value_hdd", "Delta_Full_hdd", "Delta_Sub_hdd",
              "Sum_Value_cdd", "Delta_Full_cdd", "Delta_Sub_cdd"]]
y = merge_df.vmap

parameters = {"splitter": ["best", "random"],
              "max_depth": [1, 2, 3, 5, 7, 9, 11, 12],
              "min_samples_leaf": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
              "min_weight_fraction_leaf": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
              "max_features": ["auto", "log2", "sqrt", None],
              "max_leaf_nodes": [None, 10, 20, 30, 40, 50, 60, 70, 80, 90]}
dt = tree.DecisionTreeRegressor(random_state=42)

grid_search = GridSearchCV(dt,
                           param_grid=parameters,
                           scoring='neg_mean_squared_error',
                           n_jobs=-1,
                           cv=5,
                           verbose=1)

grid_search.fit(X, y)
dt_best = grid_search.best_estimator_

score_df = pd.DataFrame(grid_search.cv_results_)
score_df.head()

viz = dtreeviz(dt_best,
               X,
               y,
               target_name='vmap',
               histtype='bar',
               feature_names=X.columns.to_list(),
               )
viz.view()


def CreateBalancedSampleWeights(y_train, largest_class_weight_coef):
    classes = y_train.unique()
    classes.sort()
    class_samples = np.bincount(y_train)
    total_samples = class_samples.sum()
    n_classes = len(class_samples)
    weights = total_samples / (n_classes * class_samples * 1.0)
    class_weight_dict = {key: value for (key, value) in zip(classes, weights)}
    class_weight_dict[classes[1]] = class_weight_dict[classes[1]
                                                      ] * largest_class_weight_coef
    sample_weights = [class_weight_dict[y] for y in y_train]

    return sample_weights


X_train, X_test, y_train, y_test = train_test_split(
    X, y, train_size=0.6, random_state=50)
X_train.shape, X_test.shape

train_sample_weight = CreateBalancedSampleWeights(
    y_train, largest_class_weight_coef=0.8)

X = merge_df[["Sum_Value_hdd", "Delta_Full_hdd", "Delta_Sub_hdd",
              "Sum_Value_cdd", "Delta_Full_cdd", "Delta_Sub_cdd"]]
y = merge_df.label

lc = LabelEncoder()
lc = lc.fit(y)
lc_y = lc.transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, lc_y, train_size=0.8, random_state=42)
X_train.shape, X_test.shape


classes_weights = class_weight.compute_sample_weight(
    class_weight='balanced',
    y=y_train
)

xgb_model = xgb.XGBClassifier(objective="multi:softprob")

params = {
    "colsample_bytree": uniform(0.7, 0.3),
    "gamma": uniform(0, 0.7),
    "learning_rate": uniform(0.03, 0.3),  # default 0.1
    "max_depth": randint(2, 20),  # default 3
    "n_estimators": randint(100, 120),  # default 100
    "subsample": uniform(0.5, 0.9)
}

search = RandomizedSearchCV(xgb_model,
                            param_distributions=params,
                            random_state=42,
                            n_iter=200,
                            cv=5,
                            verbose=0,
                            n_jobs=-1,
                            return_train_score=True)

search.fit(X_train, y_train, sample_weight=classes_weights)

report_best_scores(search.cv_results_, 1)

xgb_best = search.best_estimator_
y_pred = xgb_best.predict(X_test)

accuracy = accuracy_score(y_pred, y_test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))

cm = confusion_matrix(y_test, y_pred,)
cm_display = ConfusionMatrixDisplay(cm).plot()

confusion_matrix(y_test, y_pred)

plot_tree(xgb_best, num_trees=1)
fig = plt.gcf()
fig.set_size_inches(30, 15)


xgb.plot_importance(xgb_best)

params = {
    'max_depth': [3, 4, 5, 6, 7, 8, 10, 25, 30, 35, 40],
    'min_samples_leaf': [5, 7, 10, 15, 20, 50, 100],
    'criterion': ["gini", "entropy"],
    "max_features": ["auto", "log2", "sqrt", None]
}
dt = tree.DecisionTreeClassifier(random_state=42)


grid_search = GridSearchCV(estimator=dt,
                           param_grid=params,
                           cv=5,
                           n_jobs=-1,
                           verbose=1,
                           scoring='f1_micro')
grid_search.fit(X, y, sample_weight=classes_weights)
dt_best = grid_search.best_estimator_
report_best_scores(grid_search.cv_results_, 1)
score_df = pd.DataFrame(grid_search.cv_results_)
y_trans = y.replace({'surge': 2, 'normal': 1, 'plunge': 0})
viz = dtreeviz(dt_best,
               X,
               y_trans,
               target_name='label',
               feature_names=X.columns.to_list(),
               class_names=['plunge', 'normal', 'surge'],
               fancy=False,)


train_data = merge_df[["Sum_Value_hdd", "Delta_Full_hdd", "Delta_Sub_hdd",
                       "Sum_Value_cdd", "Delta_Full_cdd", "Delta_Sub_cdd", 'label']]

train_data, eval_data = train_test_split(train_data, test_size=0.2)

experiment = make_experiment(train_data=train_data,
                             target='label',
                             class_balance='ClassWeight',
                             cv=True,
                             num_folds=5,
                             ensemle_size=10)
estimator = experiment.run()
print(estimator)
pred = estimator.predict(eval_data.iloc[:, :-1])
test = eval_data.iloc[:, -1]
accuracy = accuracy_score(pred, test)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
cm = confusion_matrix(test, pred, labels=['plunge', 'normal', 'surge'])
cm_display = ConfusionMatrixDisplay(cm).plot()
