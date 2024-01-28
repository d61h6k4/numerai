import json

import pandas as pd
from datasets import Dataset, load_from_disk


def get_features(version: str, collection: str) -> set[str]:
    # read the metadata and display
    feature_metadata = json.load(open(f"data/v{version}/features.json"))
    return set(feature_metadata["feature_sets"][collection])


def get_targets(version: str) -> set[str]:
    # read the metadata and display
    feature_metadata = json.load(open(f"data/v{version}/features.json"))
    assert "target" in feature_metadata["targets"]
    return set(["target"])


def get_dataset(
    split: str = "train",
    version: str = "4.3",
    collection: str = "small",
    device: str = "cpu",
):
    try:
        ds = load_from_disk(f"numerai_{version}_{split}")
    except FileNotFoundError:
        data_type = split
        if split == "test":
            data_type = "test"
            split = "validation"

        df = pd.read_parquet(f"data/v{version}/{split}_int8.parquet")

        targets = get_targets(version)
        features = get_features(version, collection)
        columns = ["era"] + list(targets) + list(features)

        ds = Dataset.from_pandas(
            df[df["data_type"] == data_type][columns], split=split
        ).map(
            lambda ex: {
                "target": int(ex["target"] * 4),
            },
        )
        ds.save_to_disk(f"numerai_{version}_{split}")

    return ds.with_format("torch", device=device)
