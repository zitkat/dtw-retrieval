#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from pathlib import Path

from torch.utils.data import Dataset
import pandas as pd


class RetrievalDataset(Dataset):

    def __init__(self, samples_list_path: Path, samples_data_path: Path = None,):
        self.all_samples = pd.read_csv(samples_list_path).reset_index(drop=True)
        self.base_dir = Path(samples_list_path.parent if samples_data_path is None else samples_data_path)

    def __getitem__(self, item):
        label, file_name, freq = self.all_samples.iloc[item][["label", "file_name", "freq"]]
        return file_name, label, freq

    def __len__(self):
        return len(self.all_samples)


