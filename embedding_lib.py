#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from pathlib import Path

import h5py
import numpy as np
import torch
import tqdm

from retrieval_lib import dtwn_distance, dtwn_distance_parallel


class ActivationLoader:

    def __init__(self, input_folder: Path, model_name, checkpoint_path, layer,
                          feature_masking, positional_encoding, device=None):

        self.device = device

        self.feature_masking = feature_masking

        self.checkpoint_name = Path(checkpoint_path).stem

        self.h5path = input_folder / f"{model_name}_{layer}_{self.checkpoint_name}.h5"

        self.f = h5py.File(self.h5path, 'r')  # rework to close the file

    def load(self, name):
        dact: h5py.Dataset = self.f[name]
        act = np.array(dact)
        if self.device is not None:
            act = torch.tensor(act, device=self.device)

        feature_width = dact.attrs["feature_width"]
        prediction_str = dact.attrs.get("prediction_str", None)

        if self.feature_masking:
            act = act[..., :feature_width]

        return prediction_str, act

    def close(self):
        self.f.close()


def compute_embeddings(dataload, distance, emb, pool):
    embeddings = []
    predictions = []

    for ii, (file_name, label, freq) in tqdm.tqdm(enumerate(dataload),
                                                  total=len(dataload),
                                                  ncols=100, desc="Embedding"):
        prediction, embedding = emb.load(file_name[0])

        predictions.append(prediction)

        batch_embedding = pool(embedding)

        for bb in range(batch_embedding.shape[0]):
            embeddings.append(batch_embedding[bb, ...])

    emb.close()
    distance_matrix = distance(embeddings)

    # d1 = dtwn_distance(embeddings[:100])
    # d2 = dtwn_distance_parallel(embeddings[:100])
    # print(np.sum(d1 - d2))
    return distance_matrix, embeddings, predictions


def get_pool(pooling_strategy, **poolopts):
    if pooling_strategy == "adaptive_pool":
        return get_adaptive_pool(**poolopts)
    else:
        return globals()[pooling_strategy]


def wh_average_pool(x: torch.Tensor) -> torch.Tensor:
    dims = tuple(range(len(x.shape)))
    dims = dims[2:]

    return torch.std_mean(x, dim=dims)[1]


def id_pool(x: torch.Tensor) -> torch.Tensor:
    return x


def average_pool(*args, **kwargs):
    return torch.std_mean(*args, **kwargs)[1]


def max_pool(*args, **kwargs):
    return torch.max(*args, **kwargs).values


def adaptive_pool(x: torch.Tensor, output_length: int = 5, poolop=max_pool):
    assert output_length % 2 > 0, "output_channels must be odd"

    n_first_pools = output_length // 2 + 1
    n_second_pools = output_length - n_first_pools

    pool_size = int(np.ceil(x.shape[-1] / (output_length // 2 + 1)))

    pool_stagger = pool_size // 2

    output = torch.zeros(x.shape[:-1] + (output_length,), device=x.device)

    for ii, pool_start in enumerate(range(0, x.shape[-1], pool_size)):
        output[..., 2 * ii] = poolop(x[...,   pool_start : pool_start + pool_size], dim=-1)

    for ii, pool_start in enumerate(range(pool_stagger, x.shape[-1] - pool_size, pool_size)):
        output[..., 2*ii + 1] = poolop(x[..., pool_start : pool_start + pool_size], dim=-1)

    return torch.flatten(output, start_dim=1)


def get_adaptive_pool(target_length: int = 3, poolop: str = "average_pool"):

    def pooler(x):
        return adaptive_pool(x, output_length=target_length, poolop=globals()[poolop])

    return pooler
