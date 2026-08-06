"""Tests for the preregistration SHA-256 mechanism."""
import json
import numpy as np
import pytest

from preflight.core.preregister import (
    DatasetSpec,
    compute_preregistration_sha,
    load_preregistration,
    save_preregistration,
    verify_preregistration,
)
from preflight.core.modules.module1_grassmannian import transportability


class TestComputeSHA:
    def test_deterministic_same_inputs(self):
        spec = DatasetSpec(name="test", source_description="s", target_description="t")
        hp = {"k": 10, "metric": "geodesic"}
        r1 = compute_preregistration_sha([transportability], hp, spec)
        r2 = compute_preregistration_sha([transportability], hp, spec)
        assert r1.sha256 == r2.sha256

    def test_different_hyperparameters_different_sha(self):
        spec = DatasetSpec(name="test", source_description="s", target_description="t")
        r1 = compute_preregistration_sha([transportability], {"k": 10}, spec)
        r2 = compute_preregistration_sha([transportability], {"k": 15}, spec)
        assert r1.sha256 != r2.sha256

    def test_different_dataset_spec_different_sha(self):
        hp = {"k": 10}
        s1 = DatasetSpec(name="data_v1", source_description="s", target_description="t")
        s2 = DatasetSpec(name="data_v2", source_description="s", target_description="t")
        r1 = compute_preregistration_sha([transportability], hp, s1)
        r2 = compute_preregistration_sha([transportability], hp, s2)
        assert r1.sha256 != r2.sha256

    def test_records_module_names(self):
        spec = DatasetSpec(name="test", source_description="s", target_description="t")
        r = compute_preregistration_sha([transportability], {"k": 10}, spec)
        assert any("transportability" in m for m in r.modules_used)

    def test_records_code_lengths(self):
        spec = DatasetSpec(name="test", source_description="s", target_description="t")
        r = compute_preregistration_sha([transportability], {"k": 10}, spec)
        assert len(r.module_code_lengths) > 0
        assert all(v > 0 for v in r.module_code_lengths.values())

    def test_sha_is_64_hex_chars(self):
        spec = DatasetSpec(name="test", source_description="s", target_description="t")
        r = compute_preregistration_sha([transportability], {"k": 10}, spec)
        assert len(r.sha256) == 64
        assert all(c in "0123456789abcdef" for c in r.sha256)


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        spec = DatasetSpec(name="test", source_description="s", target_description="t")
        record = compute_preregistration_sha([transportability], {"k": 10}, spec)
        path = save_preregistration(record, tmp_path / "prereg.json")
        loaded = load_preregistration(path)
        assert loaded.sha256 == record.sha256
        assert loaded.modules_used == record.modules_used
        assert loaded.hyperparameters == record.hyperparameters

    def test_saved_file_is_valid_json(self, tmp_path):
        spec = DatasetSpec(name="test", source_description="s", target_description="t")
        record = compute_preregistration_sha([transportability], {"k": 10}, spec)
        path = save_preregistration(record, tmp_path / "prereg.json")
        with open(path) as f:
            data = json.load(f)
        assert "sha256" in data
        assert "timestamp" in data


class TestVerify:
    def test_matching_config_passes(self):
        spec = DatasetSpec(name="test", source_description="s", target_description="t")
        hp = {"k": 10}
        record = compute_preregistration_sha([transportability], hp, spec)
        matches, msg = verify_preregistration(record, [transportability], hp, spec)
        assert matches
        assert "match" in msg.lower()

    def test_changed_hyperparameters_fails(self):
        spec = DatasetSpec(name="test", source_description="s", target_description="t")
        record = compute_preregistration_sha([transportability], {"k": 10}, spec)
        matches, msg = verify_preregistration(record, [transportability], {"k": 15}, spec)
        assert not matches
        assert "hyperparameters" in msg.lower()

    def test_changed_dataset_fails(self):
        hp = {"k": 10}
        s1 = DatasetSpec(name="data_v1", source_description="s", target_description="t")
        s2 = DatasetSpec(name="data_v2", source_description="s", target_description="t")
        record = compute_preregistration_sha([transportability], hp, s1)
        matches, msg = verify_preregistration(record, [transportability], hp, s2)
        assert not matches
        assert "dataset" in msg.lower()
