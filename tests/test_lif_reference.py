"""The numba LIF kernel must reproduce Brian2 spike for spike (skipped without brian2)."""

import pytest

pytest.importorskip("brian2")
pytest.importorskip("numba")

from flybrain_robot.validation.reference import compare  # noqa: E402


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_matches_brian2_spike_for_spike(seed):
    result = compare(seed)
    assert result["brian2_spikes"] > 5000, "network too quiet to be a meaningful comparison"
    assert result["spiking_neurons"] > 300
    assert result["identical"], result
