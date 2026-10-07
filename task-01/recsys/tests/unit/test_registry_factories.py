import pytest
from replay.models import AssociationRulesItemRec, ItemKNN

from recsys.models.registry import _replay_model
from recsys.models.specs import AssociationRulesSpec, ItemKnnSpec


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        (AssociationRulesSpec(kind="association_rules", min_pair_count=2), AssociationRulesItemRec),
        (ItemKnnSpec(kind="item_knn", num_neighbours=7), ItemKNN),
    ],
)
def test_replay_factory_builds_the_estimator_for_each_kind(spec, expected):
    assert isinstance(_replay_model(spec), expected)
