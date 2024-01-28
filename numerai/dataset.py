import json
from typing import Mapping

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset
from collections import OrderedDict


def get_features(version: str, collection: str) -> OrderedDict[str, int]:
    # read the metadata and display
    feature_metadata = json.load(open(f"data/v{version}/features.json"))
    return OrderedDict(
        {v: k for k, v in enumerate(feature_metadata["feature_sets"][collection])}
    )


def get_targets(version: str) -> OrderedDict[str, int]:
    # read the metadata and display
    feature_metadata = json.load(open(f"data/v{version}/features.json"))
    assert "target" in feature_metadata["targets"]
    return OrderedDict({v: k for k, v in enumerate(["target"])})


class NumeraiDataset(TensorDataset):
    def __init__(
        self,
        df: pd.DataFrame,
        features: OrderedDict[str, int],
        targets: OrderedDict[str, int],
    ):
        super().__init__(
            torch.from_numpy(df[list(features.keys())].reset_index(drop=True).values),
            torch.from_numpy(df[list(targets.keys())].reset_index(drop=True).values),
        )


def get_dataset(
    split: str = "train",
    version: str = "4.3",
    features: OrderedDict[str, int] = OrderedDict(),
    targets: OrderedDict[str, int] = OrderedDict(),
    num: None | int = None,
):
    data_type = split
    if split == "test":
        data_type = "test"
        split = "validation"

    df = pd.read_parquet(f"data/v{version}/{split}_int8.parquet")

    columns = list(targets.keys()) + list(features.keys())

    df = df[df["data_type"] == data_type][columns]
    df["target"] = (df["target"] * 4).astype(int)
    df[list(features.keys())] = df[list(features.keys())].astype(int)
    df[list(targets.keys())] = df[list(targets.keys())].astype(np.int64)

    if num is not None:
        df = df.sample(n=num)

    return NumeraiDataset(df, features, targets)
