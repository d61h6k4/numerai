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
    def __init__(self, df: pd.DataFrame):
        self._df = df.reset_index()

    def __len__(self):
        return self._df.shape[0]

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:
        return self._df.iloc[index].to_dict()


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

    return NumeraiDataset(df)
