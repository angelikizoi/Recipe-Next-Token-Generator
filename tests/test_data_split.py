"""The train/val split must come from a single fixed permutation.

Regression: load_data drew a fresh np.random.permutation on every call. After a
single np.random.seed() in run_training, the val split was the tail of a
*different* permutation than the train split, so ~90% of val recipes were also
trained on.
"""
import numpy as np
import pytest

from config.data_config import DataConfig
from config.model_config import ModelConfig
from tokenizer.tokenizer import get_pad_idx
from training.train import load_data


def split_indices(mode):
    loader = load_data(ModelConfig(), DataConfig(), mode, get_pad_idx("custom"), "custom")
    return np.asarray(loader.batch_sampler.indices)


@pytest.fixture(scope="module")
def splits():
    # one seed, then both calls -- exactly what run_training does
    np.random.seed(1337)
    return split_indices("train"), split_indices("val")


def test_train_and_val_do_not_overlap(splits):
    train, val = splits
    assert np.intersect1d(train, val).size == 0


def test_train_and_val_together_cover_every_recipe(splits):
    train, val = splits
    n = len(np.load(DataConfig().offsets_filepath)) - 1
    assert np.array_equal(np.sort(np.concatenate([train, val])), np.arange(n))


def test_split_does_not_depend_on_global_rng_state():
    # the custom and hug runs, or any np.random call added before load_data,
    # must not change which recipes end up in val
    np.random.seed(0)
    first = split_indices("val")
    np.random.seed(1)
    second = split_indices("val")
    assert np.array_equal(first, second)