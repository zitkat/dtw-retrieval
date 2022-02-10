#!python
# -*- coding: utf-8 -*-

__author__ = "Tomas Zitka"
__email__ = "zitkat@kky.zcu.cz"

from typing import List

import click
from pathlib import Path

import pandas as pd


def join_tail(n: int, seq: List[str]):
    return seq[:n] + [" ".join(seq[n:])]


@click.command()
@click.argument('input_file', type=Path)
@click.option('--iam', 'is_iam', is_flag=True)
@click.option('--min-frequency', type=int, default=2)
@click.option('-s', '--split-selector', default=None, type=lambda p: Path(p) if p is not None else None, help="File with selected samples")
@click.option('-w', '--word-selector', default=None, help="File with selected words")
def main(input_file: Path, min_frequency, is_iam, split_selector, word_selector):

    if is_iam:
        lines = [join_tail(8, line.strip().split(" ")) for line in open(input_file, "r", encoding="utf-8")]
        all_files = pd.DataFrame(lines,
                                 columns=["file_name", "status", "graylevel",
                                          "x", "y", "w", "h", "gram", "label"],

                                 ).set_index("file_name")
        all_files[["graylevel", "x", "y", "w", "h"]] = all_files[["graylevel", "x", "y", "w", "h"]].apply(pd.to_numeric)
        all_files["status"] = all_files["status"].map({"ok": True, "err": False}).astype(bool)
    else:
        with open(input_file, "r", encoding="utf-8") as ifile:
            all_files = pd.DataFrame(map(lambda s: (s.split()[0], " ".join(s.split()[1:])), ifile.readlines()),
                                     columns=["file_name", "label"]).set_index("file_name")

    selected_files = all_files
    if split_selector is not None:
        if is_iam:
            selected_files = [line.split(",")[0] for line in open(split_selector, "r", encoding="utf-8")]
            selected_files = all_files.loc[selected_files]
        else:
            raise NotImplementedError("Selecting by split supported only for IAM dataset")

    selected_words = None
    if word_selector:
        selected_words = list(pd.read_csv(word_selector, index_col=0).dropna()["label"])

    words_grouped = selected_files.groupby("label")
    word_frequency = pd.DataFrame(map(lambda group:
                                      {"label": group[0], "freq": len(group[1])},
                                      words_grouped)).sort_values("freq", ascending=False)

    word_frequency[word_frequency["freq"] >= 2].to_csv(input_file.parent / "word_freqs.csv")

    samples = []
    not_samples = []
    for ii, word, freq in word_frequency.itertuples():
        if len(word) > 1 and freq >= min_frequency:
            if not selected_words or word in selected_words:
                print(f"{word}; {freq}; [{', '.join(all_files.loc[words_grouped.groups[word]].index)}]")
                for row in all_files.loc[words_grouped.groups[word]].itertuples():
                    samples.append(row + (freq,))
        else:
            for row in all_files.loc[words_grouped.groups[word]].itertuples():
                not_samples.append(row + (freq,))
    print(len(samples))
    samples_df = pd.DataFrame(samples, columns=["file_name"] + list(all_files.columns) + ["freq"])
    not_samples_df = pd.DataFrame(not_samples, columns=["file_name"] + list(all_files.columns) + ["freq"])

    name = split_selector.stem if split_selector is not None else input_file.stem

    samples_df.to_csv(input_file.parent / f"{name}.retrieval_{min_frequency}.csv", index=False)
    not_samples_df.to_csv(input_file.parent / f"leftover_{name}.retrieval_{min_frequency}.csv", index=False)


if __name__ == '__main__':
    main()
