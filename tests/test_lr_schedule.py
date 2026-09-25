import pytest

from training.train import get_lr

MAX_LR = 1e-4
MIN_LR = 1e-5
WARMUP = 100
DECAY = 1000


def lr_at(it):
    return get_lr(it, WARMUP, DECAY, MIN_LR, MAX_LR)


def test_warmup_starts_near_zero_and_ramps_to_peak():
    assert lr_at(0) < MAX_LR / 50
    assert lr_at(WARMUP - 1) == pytest.approx(MAX_LR, rel=1e-2)


def test_warmup_is_monotonically_increasing():
    warmup_lrs = [lr_at(i) for i in range(WARMUP)]
    assert warmup_lrs == sorted(warmup_lrs)


def test_cosine_decay_is_monotonically_decreasing():
    decay_lrs = [lr_at(i) for i in range(WARMUP, DECAY + 1)]
    assert decay_lrs == sorted(decay_lrs, reverse=True)


def test_decay_bottoms_out_at_min_lr():
    assert lr_at(DECAY) == pytest.approx(MIN_LR)


def test_lr_is_clamped_to_min_after_decay_ends():
    assert lr_at(DECAY + 1) == MIN_LR
    assert lr_at(DECAY * 10) == MIN_LR


def test_midpoint_of_cosine_is_halfway_between_min_and_max():
    midpoint = WARMUP + (DECAY - WARMUP) // 2
    assert lr_at(midpoint) == pytest.approx((MAX_LR + MIN_LR) / 2, rel=1e-2)


def test_warmup_starts_below_min_lr():
    """Warmup ramps from ~0, not from min_lr — min_lr is the decay floor only."""
    assert 0 < lr_at(0) < MIN_LR


def test_lr_stays_between_min_and_max_after_warmup():
    for it in range(WARMUP, DECAY * 2, 7):
        assert MIN_LR <= lr_at(it) <= MAX_LR
