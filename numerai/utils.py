import numpy as np
import torch
from sklearn.utils.class_weight import compute_class_weight

from numerai.dataset import get_dataset


def compute_target_weight(
    targets: set[str]
) -> dict[str, np.ndarray]:
    targets_values = {t: [] for t in targets}
    ds = get_dataset()
    for example in ds.shuffle().select(range(20_000)):
        for t in targets:
            targets_values[t].append(example[t].item())

    return {
        k: torch.tensor(
            compute_class_weight(
                class_weight="balanced",
                classes=np.unique(targets_values[k]),
                y=targets_values[k],
            ),
            dtype=torch.float,
        )
        for k in targets
    }
