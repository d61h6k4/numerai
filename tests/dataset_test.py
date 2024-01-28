from numerai.dataset import get_dataset, get_features, get_targets


def test_targets_num():
    assert len(get_targets(version="4.3")) == 1


def test_features_num():
    assert len(get_features(version="4.3", collection="small")) == 42


def test_sanity_check():
    ds = get_dataset(split="train")
    for data in ds.iter(batch_size=4):
        inputs = data["features"]
        labels = data["target"]
        print(inputs)
        print(labels)
        break
    print(ds)
