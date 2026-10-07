import math

import polars as pl

from recsys.evaluation.runner import write_results


def test_write_results_survives_metric_schema_changes_and_nan(tmp_path):
    first = pl.DataFrame(
        {"model": ["a"], "segment": ["all"], "users": [1], "HitRate@10": [float("nan")]}
    )
    second = pl.DataFrame({"model": ["b"], "segment": ["NA"], "users": [2], "HitRate@20": [0.5]})

    write_results(first, tmp_path, "val")
    paths = write_results(second, tmp_path, "val")

    merged = pl.read_parquet(paths.parquet).sort("model")
    assert merged["segment"].to_list() == ["all", "NA"]  # "NA" is not read back as null
    assert math.isnan(merged["HitRate@10"][0])
    assert merged["HitRate@20"].to_list() == [None, 0.5]
    assert paths.csv.is_file()


def test_write_results_replaces_rows_of_rerun_models(tmp_path):
    first = pl.DataFrame(
        {"model": ["a", "b"], "segment": ["all"] * 2, "users": [1, 1], "HitRate@10": [0.1, 0.2]}
    )
    rerun = pl.DataFrame({"model": ["b"], "segment": ["all"], "users": [1], "HitRate@10": [0.3]})

    write_results(first, tmp_path, "val")
    paths = write_results(rerun, tmp_path, "val")

    merged = pl.read_parquet(paths.parquet).sort("model")
    assert merged["HitRate@10"].to_list() == [0.1, 0.3]
    assert "| b | all |" in paths.markdown.read_text()


def test_write_results_uses_the_given_output_name(tmp_path):
    table = pl.DataFrame({"model": ["a"], "segment": ["all"], "users": [1], "HitRate@10": [0.1]})

    paths = write_results(table, tmp_path, "val_item_knn_sweep")

    assert paths.parquet.name == "val_item_knn_sweep.parquet"
    assert "# val_item_knn_sweep metrics" in paths.markdown.read_text()
