from pathlib import Path
from numerapi import NumerAPI

napi = NumerAPI()

version = "v4.3"

data_folder = Path("data") / version
if not data_folder.exists():
    data_folder.mkdir(parents=True)


napi.download_dataset(
    f"{version}/train_int8.parquet", str(data_folder / "train_int8.parquet")
)
napi.download_dataset(
    f"{version}/validation_int8.parquet", str(data_folder / "validation_int8.parquet")
)
napi.download_dataset(
    f"{version}/live_int8.parquet", str(data_folder / "live_int8.parquet")
)
napi.download_dataset(
    f"{version}/live_example_preds.parquet",
    str(data_folder / "live_example_preds.parquet"),
)
napi.download_dataset(
    f"{version}/validation_example_preds.parquet",
    str(data_folder / "validation_example_preds.parquet"),
)
napi.download_dataset(f"{version}/features.json", str(data_folder / "features.json"))
napi.download_dataset(
    f"{version}/meta_model.parquet", str(data_folder / "meta_model.parquet")
)
napi.download_dataset(
    f"{version}/live_benchmark_models.parquet",
    str(data_folder / "live_benchmark_models.parquet"),
)
napi.download_dataset(
    f"{version}/validation_benchmark_models.parquet",
    str(data_folder / "validation_benchmark_models.parquet"),
)
napi.download_dataset(
    f"{version}/train_benchmark_models.parquet",
    str(data_folder / "train_benchmark_models.parquet"),
)
