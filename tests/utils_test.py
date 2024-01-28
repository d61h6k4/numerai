from numerai.utils import compute_target_weight
from numerai.dataset import get_targets


def test_smoke():
    print(compute_target_weight(get_targets("4.3")))
