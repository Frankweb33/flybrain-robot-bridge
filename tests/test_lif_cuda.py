"""The GPU simulator must reproduce the CPU simulator spike for spike.

Skipped without numba-cuda-mlir or a CUDA GPU (for example on CI runners).
"""

import numpy as np
import pytest

pytest.importorskip("numba")
lif_cuda = pytest.importorskip("flybrain_robot.brain.lif_cuda")
if not lif_cuda.cuda_available():
    pytest.skip("no CUDA GPU", allow_module_level=True)

from flybrain_robot.brain.lif import LIFParams, LIFSimulator, make_simulator  # noqa: E402
from flybrain_robot.connectome.synthetic import (  # noqa: E402
    connectome_from_edges,
    random_connectome,
)


def spike_trace(simulator_class, graph, params, seed, chunks=30):
    sim = simulator_class(graph, params=params, seed=seed)
    trace = []
    for chunk in range(chunks):
        # Changing rates mid-block and uneven chunk lengths exercise partial blocks.
        sim.set_stimulus([(np.arange(0, 40), 90.0 + 7 * chunk), (np.arange(40, 60), 0.0)])
        sim.run(33.3 if chunk % 3 else 7.5)
        trace.append(sim.take_counts())
    return np.array(trace)


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gpu_matches_cpu_spike_for_spike(seed):
    # 30 % inhibitory neurons: a sign bug in GPU delivery made every neuron fire.
    graph = random_connectome(n=500, density=0.03, seed=seed)
    params = LIFParams(w_syn=1.5)
    cpu = spike_trace(LIFSimulator, graph, params, seed)
    gpu = spike_trace(lif_cuda.CudaLIFSimulator, graph, params, seed)
    assert cpu.sum() > 5000 and (cpu.sum(0) > 0).sum() > 200
    assert np.array_equal(cpu, gpu)


def test_gpu_state_matches_cpu_exactly():
    graph = connectome_from_edges(3, [0, 0, 1], [1, 2, 2], [10, -4, 6])
    params = LIFParams(w_syn=0.3)
    states = []
    for cls in (LIFSimulator, lif_cuda.CudaLIFSimulator):
        sim = cls(graph, params=params)
        sim.set_stimulus([([0], 1000 / params.dt_ms)])
        sim.run(params.dt_ms)
        sim.set_stimulus([([0], 0.0)])
        sim.run(12.3)
        states.append((np.asarray(sim.v), np.asarray(sim.g), sim.take_counts()))
    for a, b in zip(*states):
        assert np.array_equal(a, b)


def test_make_simulator_devices():
    graph = random_connectome(n=50, seed=4)
    assert isinstance(make_simulator(graph, device="cpu"), LIFSimulator)
    assert isinstance(make_simulator(graph, device="cuda"), lif_cuda.CudaLIFSimulator)
    assert isinstance(make_simulator(graph, device="auto"), lif_cuda.CudaLIFSimulator)
    with pytest.raises(ValueError, match="device"):
        make_simulator(graph, device="tpu")


def test_edge_chunks_cover_every_edge_once():
    indptr = np.array([0, 0, 3, 700, 701], np.int64)
    source, start, end = lif_cuda.edge_chunks(indptr, size=256)
    assert list(zip(source.tolist(), start.tolist(), end.tolist())) == [
        (0, 0, 0), (1, 0, 3), (2, 3, 259), (2, 259, 515), (2, 515, 700), (3, 700, 701),
    ]  # fmt: skip
