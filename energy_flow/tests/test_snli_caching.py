import os
from pathlib import Path

from energy_flow.dataset.config import DatasetConfig
from energy_flow.dataset.providers import create_snli_provider


def test_snli_cache_creation_and_limit_interplay(tmp_path, monkeypatch):
    """Ensure that:
    1. Cache file is created even when max_samples (limit) is used on first call.
    2. Second call with a different (higher) limit does not rebuild base subset.
    3. Fraction defines base subset size; limit only slices it.
    """
    # Use small fraction for speed in test environment
    cfg = DatasetConfig(
        dataset_sources=["snli"],
        snli_fraction=0.01,  # 1% base subset
        snli_min_text_length=3,
        snli_seed=123,
        snli_cache_dir=str(tmp_path / "snli_cache"),
    )

    provider = create_snli_provider(cfg, teacher_provider=None)

    # First retrieval with a limit smaller than base subset
    limit1 = 50
    pairs1 = provider.get_text_pairs(max_samples=limit1)
    assert pairs1, "Expected some SNLI pairs"
    assert len(pairs1) <= limit1

    cache_files = list((tmp_path / "snli_cache").glob("*.jsonl"))
    assert cache_files, "Cache file was not created on first limited load"
    cache_file = cache_files[0]
    cached_size = sum(1 for _ in cache_file.open('r', encoding='utf-8'))
    # Base subset should be >= first limit (can't assert exact due to filtering)
    assert cached_size >= len(pairs1)

    # Second retrieval with larger limit should return more (unless base subset smaller)
    limit2 = limit1 * 2
    pairs2 = provider.get_text_pairs(max_samples=limit2)
    assert len(pairs2) <= limit2
    # If base subset size allows, we expect more pairs now
    if cached_size > len(pairs1):
        assert len(pairs2) > len(pairs1)

    # Third retrieval without limit should return full cached base subset
    full_pairs = provider.get_text_pairs(max_samples=None)
    assert len(full_pairs) == cached_size

    # Ensure deterministic order with seed (subsequent full fetch matches)
    full_pairs_repeat = provider.get_text_pairs()
    assert [p[0] for p in full_pairs] == [p[0] for p in full_pairs_repeat]
