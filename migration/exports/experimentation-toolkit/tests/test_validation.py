import pytest

from experimentation_toolkit.validation import sample_ratio_mismatch, validate_binary_metric


def test_sample_ratio_mismatch_rejects_single_group() -> None:
    with pytest.raises(ValueError, match="at least two groups"):
        sample_ratio_mismatch({"control": 10})


def test_validate_binary_metric_rejects_non_binary_values() -> None:
    with pytest.raises(ValueError, match="binary metric"):
        validate_binary_metric([0, 1, 2])
