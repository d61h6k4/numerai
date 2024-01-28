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
        self._data = []
        for index, row in tqdm(df.iterrows(), desc="Processing dataset"):
            ex = {"id": index}
            for fn in features:
                ex[fn] = torch.tensor(row[fn], dtype=torch.int, device=device)
            for t in targets:
                ex[t] = torch.tensor(row[t], dtype=torch.long, device=device)
            self._data.append(ex)

    def __len__(self):
        return len(self._data)

    def __getitem__(self, index) -> Mapping[str, torch.Tensor]:
        return self._data[index]


def get_dataset(
    split: str = "train",
    version: str = "4.3",
    collection: str = "small",
    device: str = "cpu",
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

    return NumeraiDataset(df, features, targets, device)
