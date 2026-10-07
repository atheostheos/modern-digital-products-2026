from datetime import datetime

import pytest

from recsys.config import load_config
from recsys.data.pipeline import prepare
from recsys.evaluation.runner import TrialBudget, load_stage


@pytest.fixture
def cfg(toy_project):
    config = load_config(toy_project / "config.yaml").resolve(toy_project)
    prepare(config)
    return config


def test_full_budget_keeps_everything(cfg):
    stage = load_stage(cfg, "test")

    assert stage.window.ground_truth["user_id"].n_unique() == 2
    assert stage.train.interactions.height == 3  # train (2) + val (1)


def test_user_sample_shrinks_ground_truth_reproducibly(cfg):
    a = load_stage(cfg, "test", TrialBudget(sample_users=1))
    b = load_stage(cfg, "test", TrialBudget(sample_users=1))

    assert a.window.ground_truth["user_id"].n_unique() == 1
    assert a.window.ground_truth.equals(b.window.ground_truth)
    assert a.train.interactions.height == 3  # history is not sampled


def test_sample_larger_than_population_keeps_all_users(cfg):
    stage = load_stage(cfg, "test", TrialBudget(sample_users=1000))

    assert stage.window.ground_truth["user_id"].n_unique() == 2


def test_train_days_keeps_only_recent_history(cfg):
    # val starts 2022-10-01; 200 days back = 2022-03-15 -> only u2's 2022-04-01 review stays
    stage = load_stage(cfg, "val", TrialBudget(train_days=200))

    history = stage.train.interactions
    assert history["timestamp"].min() >= datetime(2022, 3, 15)
    assert history["user_id"].to_list() == ["u2"]
    assert stage.window.history.equals(history.select("user_id", "item_id"))


def test_budget_rejects_non_positive_values():
    with pytest.raises(ValueError, match="positive"):
        TrialBudget(sample_users=0)
    with pytest.raises(ValueError, match="positive"):
        TrialBudget(train_days=-1)
