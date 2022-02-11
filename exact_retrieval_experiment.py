#!python
# -*- coding: utf-8 -*-
"""Experiment for exact retrieval algorithm"""

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

import click
from pathlib import Path

import numpy as np
import torch

from matplotlib import pyplot as plt

from torch.utils.data import DataLoader

from util import make_path

from retrieval_dataset import RetrievalDataset

from retrieval_lib import plot_normalized_heatmap, \
    test_retrieval, get_precise_retriever, get_distance
from embedding_lib import compute_embeddings, get_pool, ActivationLoader

from settings import load_settings


@click.command()
@click.option('-i', '--input_file', type=Path,
              help="retrieval_samples.csv file")
@click.option('-s', '--settings', 'settings_path', type=Path,
              help="Settings file path.")
@click.option('-sv', '--settings-version', type=str, default="Default",
              help="Name of the settings version to use")
@click.option('-o', '--output', 'output_path', default=".outputs", type=Path)
@click.option('-c', '--checkpoint', type=Path,
              help="Path to model checkpoint, overrides checkpoint from settings!")
@click.option('--device', default=None)
@click.option('--show-plot', is_flag=False, help="Show distance matrix plot at the end.")
@click.option('--save-plot', is_flag=True,  help="Save distance matrix plot at the end.")
def main(input_file: Path,
         settings_path: Path,
         settings_version: str,
         output_path: Path,
         checkpoint: Path,
         device: str,
         show_plot: bool, save_plot: bool):
    """Runs experiment with exact nearest neighbours algorithm for different setups."""

    settings = load_settings(settings_path, settings_version)

    if checkpoint is not None:
        settings["embedding"]['checkpoint_path'] = checkpoint

    if device == "no-torch" or not settings["distance"]["torch"]:
        device = None
    else:
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_full_name = (f"{input_file.parent.name}" +
                        (f"_{input_file.stem.split('_')[2]}"
                         if len(input_file.stem.split('_')) > 2 else "") +
                        f"_{settings['embedding']['model_name']}_" +
                        f"{Path(settings['embedding']['checkpoint_path']).stem}_"
                        f"{settings_version}"
                        )

    output_full_path = make_path(output_path, output_full_name, isdir=True)

    batch_size = 1

    dataset = RetrievalDataset(input_file)

    dataload = DataLoader(dataset, batch_size=batch_size)

    pool = get_pool(**settings["pooling"])
    distance = get_distance(**settings["distance"])
    embed = ActivationLoader(input_folder=input_file.parent, **settings["embedding"], device=device)

    distance_matrix, embeddings, predictions = compute_embeddings(dataload, distance, embed, pool)

    if device is not None:
        distance_matrix = distance_matrix.cpu().numpy()
        embeddings = [emb.cpu().numpy() for emb in embeddings]

    retriever = get_precise_retriever(distance_matrix)

    ap_df, answers, hits = test_retrieval(dataset, retriever)

    # %% Outputs
    ap_df["predictions"] = np.array(predictions)[ap_df.index]

    output_folder = output_path / output_full_name

    ap_df.to_csv(make_path(output_folder, output_full_name + "_aps.csv"))
    np.save(make_path(output_folder, output_full_name + "_ans.npy"), answers)
    np.save(make_path(output_folder, output_full_name + "_hits.npy"), hits)
    np.save(make_path(output_folder, output_full_name + "_dist.npy"), distance_matrix)
    if all(embedding.shape == embeddings[0].shape for embedding in embeddings):
        np.save(make_path(output_folder, output_full_name + "_emb.npy"), np.stack(embeddings, 0))

    print(f"mAP was {ap_df.ap.mean():0.5}")

    if show_plot or save_plot:
        plot_normalized_heatmap(distance_matrix, dataset.all_samples.label)
        if save_plot:
            plt.savefig(make_path(output_folder, output_full_name + "_dist.jpg"))
            if not show_plot:
                plt.close()
        if show_plot:
            plt.show()


if __name__ == '__main__':
    main()
