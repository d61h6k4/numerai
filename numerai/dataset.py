import json
from typing import Mapping

import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


def get_features(version: str, collection: str) -> set[str]:
    # read the metadata and display
    feature_metadata = json.load(open(f"data/v{version}/features.json"))
    return set(feature_metadata["feature_sets"][collection])


def get_targets(version: str) -> set[str]:
    # read the metadata and display
    feature_metadata = json.load(open(f"data/v{version}/features.json"))
    assert "target" in feature_metadata["targets"]
    return set(["target"])


class NumeraiDataset(Dataset):
    def __init__(
        self, df: pd.DataFrame, features: set[str], targets: set[str], device: str
    ):
        self._df = df.reset_index()
        self._features = features
        self._targets = targets
        self._device = device

    def __len__(self):
        return self._df.shape[0]

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:
        raw_ex = self._df.iloc[index].to_dict()

        ex = {"id": raw_ex["id"]}
        for fn in self._features:
            ex[fn] = torch.tensor(raw_ex[fn], dtype=torch.int, device=self._device)
        for t in self._targets:
            ex[t] = torch.tensor(raw_ex[t], dtype=torch.long, device=self._device)

        return ex


def get_dataset(
    split: str = "train",
    version: str = "4.3",
    collection: str = "small",
    device: str = "cpu",
    num: None | int = None,
):
    data_type = split
    if split == "test":
        data_type = "test"
        split = "validation"

    df = pd.read_parquet(f"data/v{version}/{split}_int8.parquet")

    targets = get_targets(version)
    features = get_features(version, collection)
    columns = ["era"] + list(targets) + list(features)

    df = df[df["data_type"] == data_type][columns]
    df["target"] = (df["target"] * 4).astype(int)

    if num is not None:
        df = df.sample(n=num)

    return NumeraiDataset(df, features, targets, device)
