"""Local SparkSession for RePlay models, registered as RePlay's active session."""

import os
import sys
from collections.abc import Iterator
from contextlib import contextmanager

from pyspark.sql import SparkSession
from replay.utils.session_handler import State

from recsys.config import SparkConfig

APP_NAME = "recsys"
LOG_LEVEL = "ERROR"
RUNTIME_SQL_CONF = {
    "spark.sql.session.timeZone": "UTC",  # our timestamps are naive UTC
    "spark.sql.execution.arrow.pyspark.enabled": "true",
}


def spark_session(cfg: SparkConfig) -> SparkSession:
    """Create (or reuse) a local session in UTC and make RePlay use it.

    The caller owns the session and must ``stop()`` it when done.
    """
    # executors must run the same interpreter as the driver (the uv venv)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    cfg.local_dir.mkdir(parents=True, exist_ok=True)
    session = (
        SparkSession.builder.appName(APP_NAME)
        .master("local[*]")
        .config("spark.driver.memory", f"{cfg.driver_memory_gb}g")
        .config("spark.sql.shuffle.partitions", str(cfg.shuffle_partitions))
        .config("spark.local.dir", str(cfg.local_dir))
        .config("spark.driver.bindAddress", "127.0.0.1")
        .config("spark.driver.host", "localhost")
        .config("spark.ui.enabled", "false")
        .getOrCreate()
    )
    # runtime SQL settings are (re)applied so a reused session cannot keep other values
    for key, value in RUNTIME_SQL_CONF.items():
        session.conf.set(key, value)
    session.sparkContext.setLogLevel(LOG_LEVEL)
    State(session)
    return session


@contextmanager
def open_spark(cfg: SparkConfig) -> Iterator[SparkSession]:
    """``spark_session`` that is stopped when the block exits (CLI runs)."""
    session = spark_session(cfg)
    try:
        yield session
    finally:
        session.stop()
