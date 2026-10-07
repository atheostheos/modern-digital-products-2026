import polars as pl
import pytest
from replay.preprocessing import LabelEncoder, LabelEncodingRule

from recsys.models.replay_data import decode_items, encode_users

FRAME = pl.DataFrame({"user_id": ["u1", "u2"], "item_id": ["a", "b"]})
ENCODER = LabelEncoder([LabelEncodingRule(c) for c in ("user_id", "item_id")]).fit(FRAME)


def test_encode_users_drops_unknown_users():
    codes = encode_users(ENCODER, pl.Series(["u2", "ghost", "u2"]))

    assert codes.to_list() == [ENCODER.mapping["user_id"]["u2"]]


def test_decode_items_restores_string_ids():
    codes = pl.DataFrame({"item_id": [ENCODER.mapping["item_id"]["b"]]})

    assert decode_items(ENCODER, codes)["item_id"].to_list() == ["b"]


def test_decode_items_fails_clearly_on_unknown_codes():
    with pytest.raises(ValueError, match="unknown item codes"):
        decode_items(ENCODER, pl.DataFrame({"item_id": [999]}))
