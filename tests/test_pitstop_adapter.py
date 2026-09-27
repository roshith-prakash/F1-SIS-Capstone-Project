"""
tests/test_pitstop_adapter.py
=============================
Unit tests for PitstopAdapter, covering parametric estimation, circuit resolution,
team offsets, caution state adjustments, sampling, and performance.
"""

import time
import numpy as np
import pytest

from src.pitstop.adapter import PitstopAdapter, resolve_circuit, resolve_team


def test_adapter_initialization_and_params():
    adapter = PitstopAdapter()
    assert adapter.global_mean > 15.0
    assert adapter.global_std > 0.0
    assert "Italy" in adapter.circuit_params
    assert "Great_Britain" in adapter.circuit_params
    assert "Red Bull Racing" in adapter.team_offsets


def test_adapter_fallback_on_missing_file(tmp_path):
    missing_file = tmp_path / "non_existent_params.json"
    adapter = PitstopAdapter(params_path=missing_file)
    assert adapter.global_mean == 24.3
    assert adapter.global_std == 4.5
    mean, std = adapter.get_pit_stats("Monza", "Red Bull Racing")
    assert 15.0 <= mean <= 55.0
    assert std >= 1.0


def test_circuit_name_resolution():
    assert resolve_circuit("Monza") == "Italy"
    assert resolve_circuit("Italian Grand Prix") == "Italy"
    assert resolve_circuit("Silverstone") == "Great_Britain"
    assert resolve_circuit("British Grand Prix") == "Great_Britain"
    assert resolve_circuit("Albert Park") == "Australia"
    assert resolve_circuit("Melbourne") == "Australia"
    assert resolve_circuit("Spa-Francorchamps") == "Belgium"
    assert resolve_circuit("Unknown Venue 123") == "Unknown_Venue_123"
    assert resolve_circuit(None) == "Unknown"


def test_team_name_resolution():
    assert resolve_team("Red Bull Racing") == "Red Bull Racing"
    assert resolve_team("Red_Bull_Racing") == "Red Bull Racing"
    assert resolve_team("Ferrari") == "Ferrari"
    assert resolve_team("Scuderia Ferrari") == "Ferrari"
    assert resolve_team("AlphaTauri") == "Racing Bulls"
    assert resolve_team("RB") == "Racing Bulls"
    assert resolve_team("Racing Bulls") == "Racing Bulls"
    assert resolve_team("Kick Sauber") == "Sauber"
    assert resolve_team("Alfa Romeo") == "Sauber"
    assert resolve_team("Haas F1 Team") == "Haas F1 Team"
    assert resolve_team("Haas") == "Haas F1 Team"


def test_circuit_hierarchy_durations():
    adapter = PitstopAdapter()
    # Melbourne has a short pit lane (~18-19s)
    melb_mean, _ = adapter.get_pit_stats("Melbourne", "Ferrari")
    # Monza has standard pit lane (~24-25s)
    monza_mean, _ = adapter.get_pit_stats("Monza", "Ferrari")
    # Silverstone has long pit lane (~29-30s)
    silver_mean, _ = adapter.get_pit_stats("Silverstone", "Ferrari")

    assert melb_mean < monza_mean < silver_mean


def test_team_offsets_direction():
    adapter = PitstopAdapter()
    rb_mean, _ = adapter.get_pit_stats("Monza", "Red Bull Racing")
    haas_mean, _ = adapter.get_pit_stats("Monza", "Haas F1 Team")
    # Red Bull should be faster than Haas
    assert rb_mean < haas_mean


def test_caution_regime_effects():
    adapter = PitstopAdapter()
    mean_green, _ = adapter.get_pit_stats("Monza", "Ferrari", is_sc=False, is_vsc=False)
    mean_sc, _ = adapter.get_pit_stats("Monza", "Ferrari", is_sc=True, is_vsc=False)
    mean_vsc, _ = adapter.get_pit_stats("Monza", "Ferrari", is_sc=False, is_vsc=True)

    assert mean_green != mean_sc or mean_green != mean_vsc


def test_sampling_bounds_and_determinism():
    adapter = PitstopAdapter()
    rng1 = np.random.RandomState(123)
    rng2 = np.random.RandomState(123)

    s1 = adapter.sample_pit_duration("Monza", "Red Bull Racing", rng=rng1)
    s2 = adapter.sample_pit_duration("Monza", "Red Bull Racing", rng=rng2)
    assert s1 == pytest.approx(s2)
    assert adapter.min_clip <= s1 <= adapter.max_clip

    # 100 samples all within bounds
    samples = [
        adapter.sample_pit_duration("Silverstone", "Williams", rng=rng1)
        for _ in range(100)
    ]
    assert all(adapter.min_clip <= s <= adapter.max_clip for s in samples)


def test_adapter_throughput():
    adapter = PitstopAdapter()
    rng = np.random.RandomState(42)
    start = time.perf_counter()
    for _ in range(10000):
        _ = adapter.sample_pit_duration("Monza", "Ferrari", rng=rng)
    elapsed = time.perf_counter() - start
    # 10,000 samples should take well under 100ms
    assert elapsed < 0.15, f"Expected < 150ms for 10k samples, took {elapsed*1000:.1f}ms"
