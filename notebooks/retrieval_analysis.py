# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Retrieval experiment analysis

# %% tags=[]
from pathlib import Path

import pandas as pd
import numpy as np
import scipy.stats as st 

# Import visualization tools
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
# %matplotlib inline
mpl.rcParams['figure.dpi'] = 100 # bigger figures, yayy!

# for local imports
import sys
sys.path.append('..')
sys.path.append('../..')

from retrieval_lib import plot_normalized_heatmap

plt.style.use('ggplot')
mean_marker = {"marker":"^","markerfacecolor":"grey", "markeredgecolor":"k"}

# %% [markdown]
# ## Select experiment parameters here

# %%
dataset = "IAM_gt"
model = "resnet50"
checkpoint = "pretrained"
layer = "res"
mask_features = True
suffix = "_gpo"

# %% pycharm={"name": "#%%\n"} jupyter={"outputs_hidden": false}
experiment_id = f"{dataset}_{model}_{checkpoint}_{layer}" + ("_mf" if mask_features else "") + suffix
experiment_id

# %%
data_folder = Path("../.outputs/") / experiment_id
print(data_folder.exists())
data_folder.resolve()


# %% [markdown]
# ## Load data

# %%
aps_df = pd.read_csv(data_folder / f"{experiment_id}_aps.csv", index_col=0).reset_index(drop=True)
max_freq = aps_df["freq"].max()
data_len = len(aps_df)

# %%
distance_matrix = np.load(data_folder / f"{experiment_id}_dist.npy")

scores = -distance_matrix[~np.eye(distance_matrix.shape[0],dtype=bool)].reshape(distance_matrix.shape[0],-1)
hits = np.load(data_folder / f"{experiment_id}_hits.npy")
answers = np.load(data_folder / f"{experiment_id}_ans.npy")

# %%
aps_df.head()

# %% [markdown]
# ## Average Precision

# %%
aps_df["ap"].describe()

# %%
sns.boxplot(x = aps_df["ap"], showmeans=True, meanprops=mean_marker)

# %% [markdown]
# ### Precision-recall curve

# %%
scores.shape

# %%
hits.shape

# %%
scores_reduced = scores[:hits.shape[0]]

# %% tags=[]
from sklearn.metrics import PrecisionRecallDisplay

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score

p, r, _ = precision_recall_curve(hits.T.ravel(), scores_reduced.T.ravel())

# %% tags=[]
display = PrecisionRecallDisplay(
    recall=r,
    precision=p,)
display.plot()
_ = display.ax_.set_title("Precision-recall micro averaged over all classes")

# %% tags=[]
# precision = dict()
# recall = dict()
# average_precision = dict()
# for ii in range(data_len):
#     precision[ii], recall[ii], _ = precision_recall_curve(hits[ii, :], scores_reduced[ii, :])
#     average_precision[ii] = average_precision_score(hits[ii, :], scores_reduced[ii, :])

# %% tags=[]
# fig, ax = plt.subplots(figsize=(12, 8))

# colors = plt.cm.viridis_r(np.linspace(0,1,data_len))

# display = PrecisionRecallDisplay(
#         recall=recall[200],
#         precision=precision[200])
# display.plot(ax=ax)

# %% [markdown]
# ## Search results visualizaiton

# %%
cutoff = min(100, len(aps_df) - 1) 

# %% [markdown]
# ### Load data

# %%
answers = np.load(data_folder / f"{experiment_id}_ans.npy")
hits = np.load(data_folder / f"{experiment_id}_hits.npy")

base_len = answers.shape[1]  # size of the "database" containing all data

# %%
max_possible_freqs = np.minimum(np.array(aps_df["freq"] - 1)[..., None], np.arange(1, base_len + 1)[None, ...])

recalls = np.cumsum(hits.astype(int), axis=1) / np.array(aps_df["freq"] - 1)[..., None]
adjusted_recalls = np.cumsum(hits.astype(int), axis=1) / max_possible_freqs

mean_recall = np.mean(recalls, axis=0)
mean_adjusted_recalls = np.mean(adjusted_recalls, axis=0)

precisions = np.cumsum(hits.astype(int), axis=1) / np.arange(1, base_len + 1)[None, ...]

# %%
# Noisy data can cause this.
np.unique(np.where(recalls > 1.0)[0])

# %% [markdown]
# ### Recall plot

# %% tags=[]
plotcutoff = min(max_freq * 2, data_len - 1)

# %%
fig, ax = plt.subplots(figsize=(12, 8))
index = np.arange(1,  min(hits.shape[-1] + 1, plotcutoff + 1))
for ii in range(0, data_len, 50):
    ax.plot(index, recalls[ii, :plotcutoff],  color="gray", alpha=.1)
    # ax.plot(index, adjusted_recalls[ii, :plotcutoff],  color="c", alpha=.1)

ax.plot(index, mean_recall[:plotcutoff], linewidth=2, color='tab:red', label="Recall")
ax.plot(index, mean_adjusted_recalls[:plotcutoff], 
        linewidth=2, linestyle='--', color='tab:red', 
        label="Adjusted Recall")

if max_freq < plotcutoff:
    ax.vlines(max_freq, 0, 1,linestyle='--', color='k', alpha=.5,  label="Max occurences")

if cutoff < plotcutoff:
    ax.vlines(cutoff, 0, 1, color='k', alpha=.5, label=f"{cutoff} results")

ax.set_ylabel("Recall")
ax.set_xlabel("# Results")
ax.legend(loc="lower right")
None

# %%
print(f"Mean Recall@{cutoff}: {np.mean(recalls[:, cutoff - 1]) : 0.3}")
print(f"Mean adjusted Recall@{cutoff}: {np.mean(adjusted_recalls[:, cutoff - 1]) : 0.3}")
print(f"Min Recall@{cutoff}: {np.min(recalls[:, cutoff - 1]) : 0.3}")
print(f"#Total Recalls@{cutoff}: {np.sum(recalls[:, cutoff - 1] == 1.0)} i.e. {np.sum(recalls[:, cutoff - 1] == 1.0) / data_len : 0.3}")
print(f"#No Recalls@{cutoff}: {np.sum(recalls[:, cutoff - 1] == 0.0)} i.e. {np.sum(recalls[:, cutoff - 1] == 0.0) / data_len : 0.3}")
print()
print(f"Top-1 Precision: {np.mean(hits[:, 0].astype(int)) : 0.3}")

# %% [markdown]
# ## Recall vs OCR performance

# %%
from Levenshtein import distance as lev_distance

# %%
aps_df["lev"] = [lev_distance(pred, gt) / len(gt) for i, pred, gt in aps_df[["predictions", "label"]].itertuples()]
lev_distances = sorted(aps_df["lev"].unique())

# %% tags=[]
len(lev_distances)

# %%
aps_df["lev"].describe()

# %%
recalls_by_lev = []
for lev in lev_distances:
    recalls_by_lev.append(np.mean(
        np.cumsum(hits[aps_df["lev"] == lev, :].astype(int), axis=1) 
        / np.array(aps_df.loc[aps_df["lev"] == lev, "freq"] - 1)[..., None],
        axis=0
    ))

# %%
fig, ax = plt.subplots(figsize=(12, 8))
index = np.arange(1,  plotcutoff + 1)
# for ii in range(0, data_len, 50):
#     ax.plot(index, recalls[ii, :plotcutoff],  color="gray", alpha=.1)
# #     ax.plot(index, adjusted_recalls[ii, :cutoff],  color="c", alpha=.1)


colors = plt.cm.viridis_r(np.linspace(0,1,len(lev_distances)))

for ii, (lev, recall_by_lev) in enumerate(zip(lev_distances, recalls_by_lev)):
    ax.plot(index, recall_by_lev[:plotcutoff],  color=colors[ii], alpha=.5)

reca = ax.plot(index, mean_recall[:plotcutoff], linewidth=2, color='tab:red')
areca = ax.plot(index, mean_adjusted_recalls[:plotcutoff], 
        linewidth=2, linestyle='--', color='tab:red')

if max_freq < plotcutoff:
    ax.vlines(max_freq, 0, 1,linestyle='--', color='k', alpha=.5)

if cutoff < plotcutoff:
    ax.vlines(cutoff, 0, 1, color='k', alpha=.5)

ax.set_ylabel("Recall")
ax.set_xlabel("# Results")
# ax.legend(loc="lower right")

from matplotlib.legend import Legend
lb = Legend(ax, [reca[0], areca[0]], 
            ['Recall', "Adjusted Recall"],
            loc='lower right', frameon=False)
ax.add_artist(lb)
# ax.legend(loc="lower center", borderaxespad=-10, ncol=5, title="Levenstein distance")
None

# %% [markdown]
# ## Distance matrix

# %%
distance_matrix = np.load(data_folder / f"{experiment_id}_dist.npy")

# %%
subsample_rat = 10
fig, ax = plot_normalized_heatmap(distance_matrix[::subsample_rat, ::subsample_rat], aps_df.label)
ax.vlines([data_len /subsample_rat], ymin=0, ymax=distance_matrix.shape[1]/subsample_rat, color="grey")
ax.hlines([data_len /subsample_rat], xmin=0, xmax=distance_matrix.shape[0]/subsample_rat, color="grey")


# %%
