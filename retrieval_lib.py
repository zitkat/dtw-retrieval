#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from typing import List

import numpy as np
import pandas as pd
import torch

import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import average_precision_score

from dtw import dtw

from retrieval_dataset import RetrievalDataset


def get_distance(distance_name: str):
    return globals()[distance_name]


def euc_distance(x: List[torch.Tensor]):
    # add and remove batch dimension so that cdist works properly
    x = torch.stack(x, 0)
    return torch.cdist(x[None, ...], x[None, ...])[0, ...]


def dtwn_distance(x: List[torch.Tensor]):
    xnpy = [xj.cpu().numpy().T for xj in x]
    distance_matrix = torch.zeros(2 * (len(x),))
    for ii, xi in tqdm.tqdm(enumerate(xnpy), total=len(x), ncols=100, desc="Distance "):
        for jj, xj in enumerate(xnpy):
            d = dtw(xj, xi)
            distance_matrix[ii, jj] = d.normalizedDistance

    return distance_matrix


def test_retrieval(dataset: RetrievalDataset, retriever):
    average_precisions = []
    answers = []
    hits = []
    for ii, label, file_name, freq in tqdm.tqdm(dataset.all_samples[["label", "file_name", "freq"]].itertuples(),
                                                total=len(dataset), ncols=100, desc="Precision"):
        if freq <= 1:
            continue

        ans = retriever(ii)
        if ans[0] != ii:
            print(f"Bad first index {ans[0]} for item {ii}: {file_name} {label}")

        ans = ans[1:]  # skip the query which should be returned first

        scores = np.arange(len(ans))[::-1]
        trues = np.array((dataset.all_samples.label == label)[ans])

        answers.append(ans)
        hits.append(trues)

        if not np.any(trues):
            ap = 0
        else:
            ap = average_precision_score(trues, scores)

        average_precisions.append((ii, file_name, label, freq, ap))

    average_precisions_df = pd.DataFrame(average_precisions,
                                         columns=["index", "file_name", "label", "freq", "ap"]).set_index("index")

    return average_precisions_df, np.array(answers, dtype=int), np.array(hits, dtype=bool)


def get_precise_retriever(distance_matrix: torch.Tensor):

    def retriever(ii):
        sample_distances = distance_matrix[ii]

        sort_index = np.argsort(sample_distances)

        return sort_index

    return retriever


def plot_normalized_heatmap(distance_matrix, labels=None):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    norm_distance_matrix = (distance_matrix - distance_matrix.min()) / (distance_matrix.max() - distance_matrix.min())

    ax = sns.heatmap(norm_distance_matrix, vmin=0, vmax=1, ax=ax)
    if (labels is not None) and len(labels) < 100:
        plt.xticks(np.arange(len(labels)) + .5, labels, rotation='vertical')
        plt.yticks(np.arange(len(labels)) + .5, labels, rotation="horizontal")
    return fig, ax
