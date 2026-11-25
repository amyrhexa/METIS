# tests/test_config_schema.py
from __future__ import annotations

from pathlib import Path

import pytest

from config.schema import (
    AnalysisTaskType,
    CVConfig,
    DatasetConfig,
    ModelZooConfig,
    PipelineConfig,
    SEMTargetConfig,
    get_preset_config,
)

FIXTURES = Path(__file__).parent / "fixtures"


def test_pipeline_config_round_trips_through_dict():
    cfg = PipelineConfig(
        dataset=DatasetConfig(
            input_path="/data/x.csv",
            target_columns=["y"],
            feature_columns=["a", "b"],
            task_type=AnalysisTaskType.REGRESSION,
        ),
        cv=CVConfig(outer_folds=7, random_state=99),
        models=ModelZooConfig(selected_models=["ridge", "lasso"]),
        sem_configs={
            "y": SEMTargetConfig(
                target="y", feature_columns=["a"], missing_data_method="listwise"
            )
        },
        run_sem=True,
        output_dir="/tmp/out",
    )
    restored = PipelineConfig.from_dict(cfg.to_dict())

    assert restored.dataset.input_path == "/data/x.csv"
    assert restored.dataset.task_type == AnalysisTaskType.REGRESSION
    assert restored.cv.outer_folds == 7
    assert restored.cv.random_state == 99
    assert restored.models.selected_models == ["ridge", "lasso"]
    assert restored.run_sem is True
    assert "y" in restored.sem_configs
    assert restored.sem_configs["y"].missing_data_method == "listwise"
    assert restored.sem_configs["y"].feature_columns == ["a"]


def test_pipeline_config_save_and_load_json_round_trip(tmp_path):
    cfg = PipelineConfig(
        dataset=DatasetConfig(target_columns=["y"], feature_columns=["a"]),
        output_dir=str(tmp_path),
    )
    path = tmp_path / "config.json"
    cfg.save_json(path)
    loaded = PipelineConfig.load_json(path)
    assert loaded.dataset.target_columns == ["y"]
    assert loaded.output_dir == str(tmp_path)


def test_task_type_survives_string_and_enum_round_trip():
    cfg = PipelineConfig(dataset=DatasetConfig(task_type=AnalysisTaskType.CLASSIFICATION))
    d = cfg.to_dict()
    assert d["dataset"]["task_type"] == "classification"  # serialized as plain string
    restored = PipelineConfig.from_dict(d)
    assert restored.dataset.task_type == AnalysisTaskType.CLASSIFICATION


def test_from_dict_ignores_unknown_keys_and_fills_defaults():
    # Forward-compatibility: a manifest with an unrecognized/extra key must
    # not crash config loading.
    data = {
        "cv": {"outer_folds": 3, "totally_unknown_future_field": 123},
        "dataset": {"target_columns": ["y"]},
    }
    cfg = PipelineConfig.from_dict(data)
    assert cfg.cv.outer_folds == 3
    assert cfg.cv.inner_folds == 3  # default preserved
    assert cfg.dataset.target_columns == ["y"]


def test_sem_target_config_from_dict_supports_both_call_styles():
    payload = {"feature_columns": ["a", "b"], "missing_data_method": "fiml"}
    cfg1 = SEMTargetConfig.from_dict("cattell_totalscore", payload)
    assert cfg1.target == "cattell_totalscore"
    assert cfg1.feature_columns == ["a", "b"]

    cfg2 = SEMTargetConfig.from_dict({**payload, "target": "other"})
    assert cfg2.target == "other"


def test_get_preset_config_smoke_test_is_small_and_fast():
    cfg = get_preset_config("smoke_test")
    assert cfg.cv.outer_folds <= 3
    assert cfg.cv.inner_folds <= 3
    assert cfg.cv.skip_shap is True
    assert cfg.models.enable_stacking is False
    assert len(cfg.models.selected_models) > 0


def test_get_preset_config_unknown_name_returns_defaults():
    cfg = get_preset_config("does_not_exist")
    assert isinstance(cfg, PipelineConfig)
    assert cfg.cv.outer_folds == CVConfig().outer_folds


def test_load_manifest_parses_a_real_production_manifest():
    """Regression guard: PipelineConfig.load_manifest must stay compatible
    with the manifest shape METIS actually writes to disk (run_manifest.json),
    not just synthetic examples."""
    manifest_path = FIXTURES / "real_run_manifest.json"
    if not manifest_path.exists():
        pytest.skip("real_run_manifest.json fixture not present")

    cfg, metadata = PipelineConfig.load_manifest(manifest_path)

    assert cfg.dataset.target_columns == ["cattell_totalscore"]
    assert "yrs_education" in cfg.dataset.feature_columns
    assert cfg.cv.outer_folds == 10
    assert cfg.cv.random_state == 42
    assert cfg.run_sem is True
    assert "cattell_totalscore" in cfg.sem_configs
    assert cfg.sem_configs["cattell_totalscore"].missing_data_method == "fiml"
    assert metadata["run_id"] == "metis-run-20260904-153703-68baf5"
