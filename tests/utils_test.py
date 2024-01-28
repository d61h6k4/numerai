from numerai.utils import compute_target_weight


def test_smoke():
    print(compute_target_weight({"target"}))
