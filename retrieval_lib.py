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
import dask.bag as db
from tqdm.dask import TqdmCallback

import time
import datetime

from retrieval_dataset import RetrievalDataset
from util import now


def get_distance(distance_name: str = "", **kwargs):
    return globals()[distance_name]


def euc_distance(x: List[torch.Tensor]):
    # add and remove batch dimension so that cdist works properly
    x = torch.stack(x, 0)
    return torch.cdist(x[None, ...], x[None, ...])[0, ...]


def dtwn_distance(xnpy: List[np.ndarray]):
    xnpy = [xj.T.copy() for xj in tqdm.tqdm(xnpy, "Transfer : ", ncols=100)]
    distance_matrix = np.zeros(2 * (len(xnpy),))
    for ii, xi in tqdm.tqdm(enumerate(xnpy), total=len(xnpy),
                            ncols=100, desc="Distance "):
        for jj in range(ii, len(xnpy)):
            d = dtw(xnpy[jj], xi)
            distance_matrix[ii, jj] = d.normalizedDistance
            distance_matrix[jj, ii] = d.normalizedDistance
    return distance_matrix


def dtwn_distance_parallel(xnpy: List[np.ndarray]):
    xnpy = [xj.T.copy() for xj in tqdm.tqdm(xnpy, "Transfer : ", ncols=100)]

    distance_matrix = np.zeros(2 * (len(xnpy),))
    seq_chunks = 5
    seq_chunk_size = len(xnpy) // seq_chunks

    def work(xi, other):
        xis = np.empty(len(other))
        for jj in range(len(other)):
            xis[jj] = dtw(other[jj], xi).normalizedDistance
        return xis

    print(f"Distance : ({now()}) ")
    tstart = time.monotonic()
    for ii, istart in tqdm.tqdm(enumerate(range(0, len(xnpy), seq_chunk_size)),
                                ncols=100, total=seq_chunks, desc="Distance ",
                                disable=True):
        for jj, jstart in enumerate(range(istart, len(xnpy), seq_chunk_size), ii):
            dm = db.from_sequence(xnpy[istart:istart + seq_chunk_size], npartitions=96)
            dm = dm.map(work, other=xnpy[jstart:jstart + seq_chunk_size])

            with TqdmCallback(ncols=100, desc=f"{ii + 1}/{seq_chunks} {jj + 1}/{seq_chunks}"):
                res = dm.compute(num_workers=8)

            distance_matrix[istart:istart + seq_chunk_size,
                            jstart:jstart + seq_chunk_size] = np.array(res)
            distance_matrix[jstart:jstart + seq_chunk_size,
                            istart:istart + seq_chunk_size] = np.array(res).T
    tend = time.monotonic()
    print(f"Distance :({now()}) duration {datetime.timedelta(seconds=tend - tstart)}")

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
