"""GPU version of the LIF simulator, same arithmetic as ``lif.LIFSimulator``.

Per block of ``t_dly`` steps the GPU runs two kernels: every neuron advances through
the block (``lif_kernel.neuron_block``, compiled for CUDA), then every neuron that
spiked scatters its synaptic input into the next block with atomic integer adds.
Two launches per block, not per 0.1 ms step, keep the Windows (WDDM) launch overhead
small. The GPU processes all neurons; the CPU simulator skips neurons at rest, which
is exactly equivalent because a resting neuron without input stays at rest.

Needs ``pip install -e ".[gpu]"`` and an NVIDIA GPU. Uses numba-cuda-mlir: NVIDIA put
numba-cuda into maintenance mode and it does not import with NumPy 2.5
(github.com/NVIDIA/numba-cuda/issues/907).
"""

import math
import warnings

import numpy as np

from . import lif_kernel
from .lif import KernelConstants, LIFParams, _kernel_array, stimulus_arrays

try:
    from numba_cuda_mlir import cuda
    from numba_cuda_mlir.cuda import libdevice
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError('The GPU backend needs numba-cuda-mlir: pip install -e ".[gpu]"') from exc

# Small test networks leave most GPU threads idle; that is expected, not a problem.
warnings.filterwarnings("ignore", message="Grid size .* under-utilization")

THREADS_PER_BLOCK = 256
EDGES_PER_CHUNK = 256

_neuron_block = cuda.jit(device=True, inline=True)(
    lif_kernel.specialize(lif_kernel.neuron_block, fma=libdevice.fma)
)


@cuda.jit
def _advance_all(
    n, s0, s1, block_start, seed, v, g, refractory_until, refractory_steps, dg, pending, bits,
    counts, prob, v_0, v_rst, v_th, a_v, b_g, c_vg, kick, inv_scale, tol, block_len, clear,
):  # fmt: skip
    i = cuda.grid(1)
    if i < n:
        if s0 == 0:
            bits[i] = 0  # the previous block's spikes have been delivered
        # A neuron at rest without Poisson drive or synaptic input in this block would
        # stay exactly at rest, so skipping it changes nothing.
        if pending[i] != 0 or v[i] != v_0 or g[i] != 0.0 or prob[i] > 0.0:
            _neuron_block(i, s0, s1, block_start, seed, v, g, refractory_until,
                          refractory_steps, dg, bits, counts, prob, v_0, v_rst, v_th, a_v,
                          b_g, c_vg, kick, inv_scale, tol, block_len)  # fmt: skip
        if clear and pending[i] != 0:  # the block is finished: its input row is used up
            row = i * block_len
            for s in range(block_len):
                dg[row + s] = 0
            pending[i] = 0


@cuda.jit
def _deliver(n_chunks, block_len, bits, chunk_source, chunk_start, chunk_end, indices,
             weight_fixed, dg_next, pending_next):  # fmt: skip
    # One thread per chunk of at most EDGES_PER_CHUNK outgoing edges, so a neuron with
    # thousands of targets is delivered by many threads instead of one.
    c = cuda.grid(1)
    if c < n_chunks:
        b = bits[chunk_source[c]]
        if b != 0:
            for s in range(block_len):
                if (b >> s) & 1:
                    for e in range(chunk_start[c], chunk_end[c]):
                        target = indices[e]
                        # The explicit int64 matters: numba-cuda-mlir 0.5.2 zero-extends an
                        # int32 value added to an int64 array, turning -3 into 4294967293.
                        cuda.atomic.add(dg_next, target * block_len + s, np.int64(weight_fixed[e]))
                        pending_next[target] = 1  # every writer stores the same value


def edge_chunks(indptr, size=EDGES_PER_CHUNK):
    """(source, start, end) for consecutive runs of at most ``size`` edges per neuron."""
    degree = np.diff(indptr)
    per_neuron = np.maximum(1, -(-degree // size))
    source = np.repeat(np.arange(len(degree), dtype=np.int32), per_neuron)
    first = np.repeat(np.cumsum(per_neuron) - per_neuron, per_neuron)
    start = indptr[source] + (np.arange(len(source)) - first) * size
    end = np.minimum(start + size, indptr[source + 1])
    return source, start.astype(np.int64), end.astype(np.int64)


def cuda_available():
    try:
        return bool(cuda.is_available())
    except Exception:  # driver or runtime problems all mean "no usable GPU"
        return False


class CudaLIFSimulator:
    """Drop-in replacement for ``LIFSimulator`` that keeps state on the GPU."""

    device = "cuda"

    def __init__(self, connectome, params=None, seed=0):
        if not cuda_available():
            raise ValueError("No CUDA GPU available")
        self.connectome = connectome
        self.params = params or LIFParams.matched_to(connectome.manifest)
        self.seed = seed
        self.n = n = connectome.size
        self.constants = k = KernelConstants(self.params, connectome)
        self.default_refractory = k.refractory
        self.launch = (math.ceil(n / THREADS_PER_BLOCK), THREADS_PER_BLOCK)
        host_indptr = _kernel_array(connectome.indptr, np.int64)
        source, start, end = edge_chunks(host_indptr)
        self.n_chunks = len(source)
        self.chunk_launch = (math.ceil(self.n_chunks / THREADS_PER_BLOCK), THREADS_PER_BLOCK)
        self.chunk_source = cuda.to_device(source)
        self.chunk_start = cuda.to_device(start)
        self.chunk_end = cuda.to_device(end)
        self.indptr = cuda.to_device(host_indptr)
        self.indices = cuda.to_device(_kernel_array(connectome.indices, np.int32))
        self.weight_fixed = cuda.to_device(k.weight_fixed)
        self._v = cuda.device_array(n, np.float64)
        self._g = cuda.device_array(n, np.float64)
        self.refractory_until = cuda.device_array(n, np.int64)
        self.host_refractory_steps = np.full(n, k.refractory, np.int64)
        self.refractory_steps = cuda.to_device(self.host_refractory_steps)
        self.dg_cur = cuda.device_array(n * k.block, np.int64)
        self.dg_next = cuda.device_array(n * k.block, np.int64)
        self.pending_cur = cuda.device_array(n, np.uint8)
        self.pending_next = cuda.device_array(n, np.uint8)
        self.bits = cuda.device_array(n, np.uint32)
        self._counts = cuda.device_array(n, np.int32)
        self.host_prob = np.zeros(n, np.float64)
        self.prob = cuda.to_device(self.host_prob)
        self.stim_index = np.zeros(0, np.int32)
        self.reset()

    def reset(self):
        for array in (self._g, self.refractory_until, self.dg_cur, self.dg_next, self.bits,
                      self._counts, self.pending_cur, self.pending_next):  # fmt: skip
            array[:] = 0
        self._v[:] = self.params.v_0
        self.step_index = 0

    @property
    def time_ms(self):
        return self.step_index * self.params.dt_ms

    @property
    def v(self):
        return self._v.copy_to_host()

    @property
    def g(self):
        return self._g.copy_to_host()

    @property
    def counts(self):
        return self._counts.copy_to_host()

    def set_stimulus(self, drive):
        """Same semantics as ``LIFSimulator.set_stimulus``."""
        new_index, prob = stimulus_arrays(drive, self.params.dt_ms, self.n)
        if not np.array_equal(new_index, self.stim_index):
            self.host_refractory_steps[self.stim_index] = self.default_refractory
            self.host_refractory_steps[new_index] = 0
            self.refractory_steps.copy_to_device(self.host_refractory_steps)
            self.stim_index = new_index
        self.host_prob[:] = prob
        self.prob.copy_to_device(self.host_prob)

    def run(self, duration_ms):
        steps = round(duration_ms / self.params.dt_ms)
        if steps < 0:
            raise ValueError("duration must be non-negative")
        k = self.constants
        step, end = self.step_index, self.step_index + steps
        while step < end:
            block_start = (step // k.block) * k.block
            s0 = step - block_start
            s1 = min(k.block, end - block_start)
            finished = s1 == k.block
            _advance_all[self.launch](
                self.n, s0, s1, block_start, self.seed, self._v, self._g,
                self.refractory_until, self.refractory_steps, self.dg_cur, self.pending_cur,
                self.bits, self._counts, self.prob, *k.scalars, k.block, finished,
            )  # fmt: skip
            if finished:
                _deliver[self.chunk_launch](self.n_chunks, k.block, self.bits, self.chunk_source,
                                            self.chunk_start, self.chunk_end, self.indices,
                                            self.weight_fixed, self.dg_next,
                                            self.pending_next)  # fmt: skip
                self.dg_cur, self.dg_next = self.dg_next, self.dg_cur
                self.pending_cur, self.pending_next = self.pending_next, self.pending_cur
            step = block_start + s1
        cuda.synchronize()
        self.step_index = end
        return steps

    def take_counts(self):
        counts = self._counts.copy_to_host()
        self._counts[:] = 0
        return counts
