from pathlib import Path

import pytest

from recsys import spark as spark_module
from recsys.config import SparkConfig

CFG = SparkConfig(driver_memory_gb=1, shuffle_partitions=1, local_dir=Path("unused"))


class FakeSession:
    def __init__(self) -> None:
        self.stopped = False

    def stop(self) -> None:
        self.stopped = True


def test_open_spark_stops_the_session_on_exit(monkeypatch):
    fake = FakeSession()
    monkeypatch.setattr(spark_module, "spark_session", lambda cfg: fake)

    with spark_module.open_spark(CFG) as session:
        assert session is fake and not fake.stopped

    assert fake.stopped


def test_open_spark_stops_the_session_on_error(monkeypatch):
    fake = FakeSession()
    monkeypatch.setattr(spark_module, "spark_session", lambda cfg: fake)

    with pytest.raises(ValueError, match="boom"), spark_module.open_spark(CFG):
        raise ValueError("boom")

    assert fake.stopped
