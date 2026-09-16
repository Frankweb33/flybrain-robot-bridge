"""Event-driven leaky integrate-and-fire simulation of a whole connectome (CPU).

Model and parameters follow Shiu et al. 2024, "A Drosophila computational brain
model reveals sensorimotor processing" (Nature 634, 210-219), which ran the same
equations on the FlyWire connectome in Brian2 (github.com/philshiu/Drosophila_brain_model):

    dv/dt = (v_0 - v + g) / t_mbr        (v and g frozen while refractory)
    dg/dt = -g / tau
    spike when v > v_th, then v = v_rst, g = 0, refractory for t_rfc
    each presynaptic spike adds weight * w_syn to g of the target after t_dly
    (dropped if the target is refractory, as Brian2 does for such variables)

``weight`` is the signed synapse count from the connectome. The linear system is
integrated exactly, like Brian2's ``method='linear'``. Sensory drive is a Poisson
train whose events add ``poisson_kick_mv`` to v; driven neurons have no refractory
period, as in the reference. A neuron within ``rest_tolerance`` mV of rest in both v
and g is set back to rest exactly.

The simulation advances in blocks of ``t_dly`` steps, inside which neurons do not
interact (see ``lif_kernel.py``). Only neurons away from rest, with pending input or
with Poisson drive are processed, in parallel when there are many, so a silent brain
costs almost nothing. ``lif_cuda.py`` runs the same arithmetic on a GPU.

This is a connectome-constrained dynamics approximation, not a biophysically
validated model of a fly.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

from . import lif_kernel

try:
    from llvmlite import ir
    from numba import njit, prange
    from numba import types as nbtypes
    from numba.extending import intrinsic
except ImportError as exc:  # pragma: no cover - exercised only without the extra
    raise ImportError('The LIF backend needs numba: pip install -e ".[connectome]"') from exc

# Shiu et al. calibrated w_syn = 0.275 mV on FlyWire v783: 54,492,922 synapses among
# 138,639 neurons (Connectivity_783.parquet / Completeness_783.csv in their repository).
SHIU_W_SYN_MV = 0.275
FLYWIRE_783_SYNAPSES_PER_NEURON = 54_492_922 / 138_639


@dataclass(frozen=True)
class LIFParams:
    dt_ms: float = 0.1
    v_0: float = -52.0
    v_rst: float = -52.0
    v_th: float = -45.0
    t_mbr_ms: float = 20.0
    tau_ms: float = 5.0
    # Synapses of cells marked slow use this time constant instead. Direction selectivity
    # in the fly needs inputs with different kinetics (Mi9 slow and sign-inverting against
    # fast Mi1/Tm3: Arenz et al. 2017; Strother et al. 2017); the connectome fixes which
    # cell contacts which, not how fast. Equal to tau_ms it changes nothing.
    tau_slow_ms: float = 5.0
    t_rfc_ms: float = 2.2
    t_dly_ms: float = 1.8
    w_syn: float = SHIU_W_SYN_MV
    poisson_kick_mv: float = SHIU_W_SYN_MV * 250
    rest_tolerance: float = 1e-4

    @classmethod
    def matched_to(cls, manifest, **overrides):
        """Rescale w_syn so mean synaptic input per neuron equals the FlyWire calibration.

        MaleCNS has more synapses per neuron than FlyWire (different reconstruction and
        synapse detection, plus the ventral nerve cord). Keeping 0.275 mV would roughly
        double the drive the free parameter was fitted to.
        """
        per_neuron = manifest["synapses"] / manifest["neurons"]
        w_syn = SHIU_W_SYN_MV * FLYWIRE_783_SYNAPSES_PER_NEURON / per_neuron
        return replace(cls(w_syn=w_syn), **overrides)


class KernelConstants:
    """Derived per-step constants, identical for the CPU and GPU simulators."""

    def __init__(self, params, connectome):
        p = params
        self.block = max(1, round(p.t_dly_ms / p.dt_ms))  # synaptic delay in steps
        if self.block > lif_kernel.MAX_BLOCK_STEPS:
            raise ValueError("t_dly_ms / dt_ms must be at most 32 steps")
        # Brian2 keeps a neuron refractory while t < t_spike + t_rfc.
        self.refractory = max(0, math.ceil(p.t_rfc_ms / p.dt_ms - 1e-9) - 1)
        self.a_v = math.exp(-p.dt_ms / p.t_mbr_ms)
        self.b_g = math.exp(-p.dt_ms / p.tau_ms)
        self.c_vg = p.tau_ms / (p.tau_ms - p.t_mbr_ms) * (self.b_g - self.a_v)
        self.b_h = math.exp(-p.dt_ms / p.tau_slow_ms)
        self.c_vh = p.tau_slow_ms / (p.tau_slow_ms - p.t_mbr_ms) * (self.b_h - self.a_v)
        self.inv_scale = 2.0**-lif_kernel.FIXED_SHIFT
        scaled = np.rint(np.asarray(connectome.weight, np.float64) * p.w_syn * 2**lif_kernel.FIXED_SHIFT)
        if scaled.size and np.abs(scaled).max() >= 2**31:
            raise ValueError("w_syn too large for 32-bit fixed-point synaptic weights")
        self.weight_fixed = _kernel_array(scaled, np.int32)
        self.scalars = (
            float(p.v_rst), float(p.v_th), self.a_v, self.b_g, self.b_h, self.c_vg, self.c_vh,
            float(p.poisson_kick_mv), self.inv_scale, float(p.rest_tolerance),
        )  # fmt: skip


PARALLEL_MIN_ACTIVE = 2_000
RESORT_MIN_ENTRANTS = 1_024

@intrinsic
def _fma(typingctx, a, b, c):
    """LLVM's fma: the FMA3 instruction where available, else libm, both correctly rounded."""
    signature = nbtypes.float64(nbtypes.float64, nbtypes.float64, nbtypes.float64)

    def codegen(context, builder, sig, args):
        double = ir.DoubleType()
        function_type = ir.FunctionType(double, [double] * 3)
        return builder.call(builder.module.declare_intrinsic("llvm.fma", [double], function_type), args)

    return signature, codegen


_neuron_block = njit(inline="always")(lif_kernel.specialize(lif_kernel.neuron_block, fma=_fma))


@njit(cache=True)
def _advance(
    active, n_active, s0, s1, block_start, seed, v, g, h, refractory_until, refractory_steps,
    dg, dh, bits, counts, prob, v_rest, v_rst, v_th, a_v, b_g, b_h, c_vg, c_vh, kick,
    inv_scale, tol, block_len,
):  # fmt: skip
    for a in range(n_active):
        _neuron_block(active[a], s0, s1, block_start, seed, v, g, h, refractory_until,
                      refractory_steps, dg, dh, bits, counts, prob, v_rest, v_rst, v_th, a_v,
                      b_g, b_h, c_vg, c_vh, kick, inv_scale, tol, block_len)  # fmt: skip


@njit(cache=True, parallel=True)
def _advance_parallel(
    active, n_active, s0, s1, block_start, seed, v, g, h, refractory_until, refractory_steps,
    dg, dh, bits, counts, prob, v_rest, v_rst, v_th, a_v, b_g, b_h, c_vg, c_vh, kick,
    inv_scale, tol, block_len,
):  # fmt: skip
    for a in prange(n_active):
        _neuron_block(active[a], s0, s1, block_start, seed, v, g, h, refractory_until,
                      refractory_steps, dg, dh, bits, counts, prob, v_rest, v_rst, v_th, a_v,
                      b_g, b_h, c_vg, c_vh, kick, inv_scale, tol, block_len)  # fmt: skip


@njit(cache=True)
def _finish_block(
    active, n_active, active_flag, entrants, indptr, indices, weight_fixed, slow, dg_cur,
    dg_next, dh_cur, dh_next, pending_cur, pending_next, bits, prob, v, g, h, v_rest, v_th,
    block_len,
):  # fmt: skip
    """Deliver this block's spikes into the next block, clear used input, retire at rest."""
    n_entrants = 0
    for a in range(n_active):
        i = active[a]
        if pending_cur[i]:
            row = i * block_len
            for s in range(block_len):
                dg_cur[row + s] = 0
                dh_cur[row + s] = 0
            pending_cur[i] = False
        b = bits[i]
        if b != 0:
            is_slow = slow[i]
            for s in range(block_len):
                if (b >> s) & 1:
                    for e in range(indptr[i], indptr[i + 1]):
                        target = indices[e]
                        if is_slow:
                            dh_next[target * block_len + s] += weight_fixed[e]
                        else:
                            dg_next[target * block_len + s] += weight_fixed[e]
                        if not pending_next[target]:
                            pending_next[target] = True
                            if not active_flag[target]:
                                active_flag[target] = True
                                entrants[n_entrants] = target
                                n_entrants += 1
            bits[i] = 0
    kept = 0
    for a in range(n_active):
        i = active[a]
        if (pending_next[i] or prob[i] > 0.0 or v[i] != v_rest[i] or g[i] != 0.0
                or h[i] != 0.0 or v_rest[i] > v_th):  # a tonically firing neuron never rests
            active[kept] = i
            kept += 1
        else:
            active_flag[i] = False
    if n_entrants >= RESORT_MIN_ENTRANTS:
        # Index order keeps memory access sequential; worth a scan when many joined.
        kept = 0
        for i in range(active_flag.shape[0]):
            if active_flag[i]:
                active[kept] = i
                kept += 1
        return kept
    for k in range(n_entrants):
        active[kept + k] = entrants[k]
    return kept + n_entrants


@njit(cache=True)
def _run(
    steps, t0, block_len, seed, indptr, indices, weight_fixed, slow, v, g, h,
    refractory_until, refractory_steps, dg_cur, dg_next, dh_cur, dh_next, pending_cur,
    pending_next, bits, counts, prob, active_flag, active, n_active, entrants,
    v_rest, v_rst, v_th, a_v, b_g, b_h, c_vg, c_vh, kick, inv_scale, tol,
):  # fmt: skip
    step = t0
    end = t0 + steps
    swaps = 0
    while step < end:
        block_start = (step // block_len) * block_len
        s0 = step - block_start
        s1 = min(block_len, end - block_start)
        if n_active >= PARALLEL_MIN_ACTIVE:
            _advance_parallel(active, n_active, s0, s1, block_start, seed, v, g, h,
                              refractory_until, refractory_steps, dg_cur, dh_cur, bits, counts,
                              prob, v_rest, v_rst, v_th, a_v, b_g, b_h, c_vg, c_vh, kick,
                              inv_scale, tol, block_len)  # fmt: skip
        else:
            _advance(active, n_active, s0, s1, block_start, seed, v, g, h, refractory_until,
                     refractory_steps, dg_cur, dh_cur, bits, counts, prob, v_rest, v_rst, v_th,
                     a_v, b_g, b_h, c_vg, c_vh, kick, inv_scale, tol, block_len)  # fmt: skip
        step = block_start + s1
        if s1 == block_len:
            n_active = _finish_block(active, n_active, active_flag, entrants, indptr, indices,
                                     weight_fixed, slow, dg_cur, dg_next, dh_cur, dh_next,
                                     pending_cur, pending_next, bits, prob, v, g, h, v_rest,
                                     v_th, block_len)  # fmt: skip
            dg_cur, dg_next = dg_next, dg_cur
            dh_cur, dh_next = dh_next, dh_cur
            pending_cur, pending_next = pending_next, pending_cur
            swaps += 1
    return n_active, swaps


def stimulus_arrays(drive, dt_ms, size):
    """(listed neuron indices, per-step Poisson probability for every neuron)."""
    index, rates = [], []
    for neurons, rate in drive:
        neurons = np.asarray(neurons, np.int32).ravel()
        rate = np.broadcast_to(np.asarray(rate, np.float64), neurons.shape)
        if not np.all(np.isfinite(rate)) or np.any(rate < 0):
            raise ValueError("Stimulus rates must be finite and non-negative")
        index.append(neurons)
        rates.append(rate)
    prob = np.zeros(size, np.float64)
    if not index:
        return np.zeros(0, np.int32), prob
    index = np.concatenate(index)
    np.add.at(prob, index, np.concatenate(rates) * dt_ms / 1000)
    return index, np.minimum(prob, 1.0, out=prob)


def resting_for_rate(params, rate_hz):
    """Resting potential at which a neuron with no input fires at about ``rate_hz``.

    Most neurons of the fly's optic lobe do not spike: they release transmitter
    continuously, and a signal is as often a drop below that baseline as a rise above
    it. A spiking neuron can only stand in for one if it fires on its own, which is what
    a resting potential above threshold does. The rate is approximate because the model
    advances in 0.1 ms steps; the achieved rate is measured, not assumed.
    """
    if rate_hz <= 0:
        return params.v_0
    charge_ms = 1000.0 / rate_hz - params.t_rfc_ms
    if charge_ms <= 0:
        raise ValueError("rate_hz above the refractory limit")
    rise = 1.0 - math.exp(-charge_ms / params.t_mbr_ms)
    return params.v_rst + (params.v_th - params.v_rst) / rise


def _kernel_array(values, dtype):
    flags = ["C_CONTIGUOUS", "WRITEABLE", "OWNDATA"]
    return np.require(np.asarray(values, dtype=dtype), dtype, flags)


def make_simulator(connectome, params=None, seed=0, device="auto", v_rest=None, slow=None):
    """CPU or GPU simulator. Both give bit-identical spikes, so "auto" is safe."""
    if device not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
    if device != "cpu":
        try:
            from .lif_cuda import CudaLIFSimulator, cuda_available
        except ImportError as exc:
            if device == "cuda":
                raise ValueError(f"device 'cuda' needs the gpu extra: {exc}") from exc
        else:
            if cuda_available():
                return CudaLIFSimulator(connectome, params=params, seed=seed, v_rest=v_rest,
                                        slow=slow)
            if device == "cuda":
                raise ValueError("device 'cuda' requested but no CUDA GPU is available")
    return LIFSimulator(connectome, params=params, seed=seed, v_rest=v_rest, slow=slow)


class LIFSimulator:
    device = "cpu"

    def __init__(self, connectome, params=None, seed=0, v_rest=None, slow=None):
        self.connectome = connectome
        self.params = params or LIFParams.matched_to(connectome.manifest)
        self.seed = seed
        n = connectome.size
        self.v_rest = _kernel_array(
            np.full(n, self.params.v_0) if v_rest is None else np.asarray(v_rest), np.float64
        )
        if self.v_rest.shape != (n,):
            raise ValueError("v_rest needs one resting potential per neuron")
        self.slow = _kernel_array(np.zeros(n) if slow is None else np.asarray(slow), np.bool_)
        if self.slow.shape != (n,):
            raise ValueError("slow needs one flag per neuron")
        # One array type per argument keeps numba to a single compiled specialization.
        # Several cached specializations of a kernel that calls a parallel function
        # crashed on load (read-only memmaps and writable arrays type differently).
        self.constants = k = KernelConstants(self.params, connectome)
        self.indptr = _kernel_array(connectome.indptr, np.int64)
        self.indices = _kernel_array(connectome.indices, np.int32)
        self.weight_fixed = k.weight_fixed
        self.default_refractory = k.refractory
        self.v = np.empty(n, np.float64)
        self.g = np.empty(n, np.float64)
        self.h = np.empty(n, np.float64)
        self.refractory_until = np.empty(n, np.int64)
        self.refractory_steps = np.full(n, k.refractory, np.int64)
        self.dg_cur = np.zeros(n * k.block, np.int64)
        self.dg_next = np.zeros(n * k.block, np.int64)
        self.dh_cur = np.zeros(n * k.block, np.int64)
        self.dh_next = np.zeros(n * k.block, np.int64)
        self.pending_cur = np.zeros(n, np.bool_)
        self.pending_next = np.zeros(n, np.bool_)
        self.bits = np.zeros(n, np.uint32)
        self.counts = np.zeros(n, np.int32)
        self.prob = np.zeros(n, np.float64)
        self.active_flag = np.zeros(n, np.bool_)
        self.active = np.empty(n, np.int32)
        self.entrants = np.empty(n, np.int32)
        self.stim_index = np.zeros(0, np.int32)
        self.reset()

    def reset(self):
        for array in (self.g, self.h, self.refractory_until, self.dg_cur, self.dg_next,
                      self.dh_cur, self.dh_next, self.bits, self.counts, self.pending_cur,
                      self.pending_next):  # fmt: skip
            array.fill(0)
        self.v[:] = self.v_rest
        self.active_flag.fill(False)
        self.n_active = 0
        self.step_index = 0
        self._activate(np.flatnonzero((self.prob > 0) | (self.v_rest > self.params.v_th)))

    @property
    def time_ms(self):
        return self.step_index * self.params.dt_ms

    def _activate(self, neurons):
        new = np.asarray(neurons, np.int32)
        new = new[~self.active_flag[new]]
        self.active_flag[new] = True
        self.active[self.n_active : self.n_active + len(new)] = new
        self.n_active += len(new)

    def set_stimulus(self, drive):
        """``drive`` is an iterable of ``(neuron_indices, poisson_rate_hz)`` pairs.

        The rate is a number or an array with one rate per listed neuron. Each neuron
        has one Poisson rate; a neuron listed twice gets the sum of its rates (a
        probability of at most 1 per step). Driven neurons have no refractory period, as
        in the reference model, until they are left out of a later stimulus.
        """
        index, prob = stimulus_arrays(drive, self.params.dt_ms, self.connectome.size)
        self.refractory_steps[self.stim_index] = self.default_refractory
        self.stim_index = _kernel_array(index, np.int32)
        self.prob[:] = prob
        self.refractory_steps[self.stim_index] = 0
        self._activate(np.flatnonzero(self.prob > 0))

    def run(self, duration_ms):
        steps = round(duration_ms / self.params.dt_ms)
        if steps < 0:
            raise ValueError("duration must be non-negative")
        k = self.constants
        n_active, swaps = _run(
            int(steps), int(self.step_index), int(k.block), int(self.seed), self.indptr,
            self.indices, self.weight_fixed, self.slow, self.v, self.g, self.h,
            self.refractory_until, self.refractory_steps, self.dg_cur, self.dg_next,
            self.dh_cur, self.dh_next, self.pending_cur, self.pending_next, self.bits,
            self.counts, self.prob, self.active_flag, self.active, int(self.n_active),
            self.entrants, self.v_rest, *k.scalars,
        )  # fmt: skip
        self.n_active = int(n_active)
        if swaps % 2:
            self.dg_cur, self.dg_next = self.dg_next, self.dg_cur
            self.dh_cur, self.dh_next = self.dh_next, self.dh_cur
            self.pending_cur, self.pending_next = self.pending_next, self.pending_cur
        self.step_index += steps
        return steps

    def take_counts(self):
        """Spike counts per neuron since the previous call."""
        counts = self.counts.copy()
        self.counts.fill(0)
        return counts
