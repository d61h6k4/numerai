import argparse
from pathlib import Path

import pandas as pd
import torch

# import the 2 scoring functions
from numerai_tools.scoring import correlation_contribution, numerai_corr
from seaborn import objects as so
import matplotlib as mpl
from tqdm import tqdm

from numerai.dataset import get_dataset
from numerai.model import NumeraiModel


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model",
        type=Path,
        help="Specify path to the model state dict.",
        required=True,
    )

    return parser.parse_args()


def main():
    args = parse_args()

    device = "mps"

    saved_model = NumeraiModel()
    saved_model.load_state_dict(torch.load(args.model))
    saved_model = saved_model.to(device)
    saved_model.eval()

    validation_loader = torch.utils.data.DataLoader(
        get_dataset(split="validation", device=device), batch_size=1024, shuffle=False
    )

    predictions = {"id": [], "prediction": []}
    # Disable gradient computation and reduce memory consumption.
    with torch.no_grad():
        for vdata in tqdm(validation_loader):
            vinputs = vdata
            voutputs = saved_model(vinputs)

            predictions["id"].extend(vdata["id"])
            predictions["prediction"].extend(
                (torch.argmax(voutputs, dim=-1) / 4.0).cpu().numpy().tolist()
            )

    predictions_df = pd.DataFrame(predictions).set_index("id")
    validation = pd.read_parquet(
        "data/v4.3/validation_int8.parquet", columns=["era", "target", "data_type"]
    )
    validation = validation[validation["data_type"] == "validation"]
    del validation["data_type"]
    validation["meta_model"] = pd.read_parquet("data/v4.3/meta_model.parquet")[
        "numerai_meta_model"
    ]
    validation = validation.join(predictions_df, how="inner")

    # Compute the per-era corr between our predictions and the target values
    per_era_corr = validation.groupby("era").apply(
        lambda x: numerai_corr(x[["prediction"]].dropna(), x["target"].dropna())
    )

    # Compute the per-era mmc between our predictions, the meta model, and the target values
    per_era_mmc = (
        validation.dropna()
        .groupby("era")
        .apply(
            lambda x: correlation_contribution(
                x[["prediction"]], x["meta_model"], x["target"]
            )
        )
    )

    f = mpl.figure.Figure(figsize=(20, 40))
    sf1, sf2, sf3, sf4 = f.subfigures(4, 1)

    (
        so.Plot(per_era_corr, x="era", y="prediction")
        .add(so.Bar())
        .label(title="Validation CORR")
        .on(sf1)
        .plot()
    )

    (
        so.Plot(per_era_mmc, x="era", y="prediction")
        .add(so.Bar())
        .label(title="Validation MMC")
        .on(sf2)
        .plot()
    )

    (
        so.Plot(per_era_corr.cumsum(), x="era", y="prediction")
        .add(so.Line())
        .label(title="Cummulative Validation CORR")
        .on(sf3)
        .plot()
    )
    (
        so.Plot(per_era_mmc.cumsum(), x="era", y="prediction")
        .add(so.Line())
        .label(title="Cummulative Validation MMC")
        .on(sf4)
        .plot()
    )
    f.savefig(f"eval_{args.model}.png")


main()
