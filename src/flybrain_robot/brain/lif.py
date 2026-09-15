"""Event-driven leaky integrate-and-fire simulation of a whole connectome.

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
period, as in the reference. Only neurons away from rest are integrated, so a
silent brain costs almost nothing; a neuron within ``rest_tolerance`` mV of rest
in both v and g is set back to rest exactly.

This is a connectome-constrained dynamics approximation, not a biophysically
validated model of a fly.
"""

import math
from dataclasses import dataclass, replace

import numpy as np

try:
    from numba import njit, prange
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


PARALLEL_MIN_ACTIVE = 20_000
REORDER_EVERY_STEPS = 20


@njit(cache=True)
def _seed(seed):
    np.random.seed(seed)


@njit(cache=True, inline="always")
def _update(i, a, v, g, refractory_until, step, v_0, v_th, a_v, b_g, c_vg, fired):
    fired[a] = False
    if step >= refractory_until[i]:
        v[i] = v_0 + (v[i] - v_0) * a_v + g[i] * c_vg
        g[i] *= b_g
        fired[a] = v[i] > v_th


@njit(cache=True)
def _integrate(active, n_active, v, g, refractory_until, step, v_0, v_th, a_v, b_g, c_vg, fired):
    for a in range(n_active):
        _update(active[a], a, v, g, refractory_until, step, v_0, v_th, a_v, b_g, c_vg, fired)


@njit(cache=True, parallel=True)
def _integrate_parallel(
    active, n_active, v, g, refractory_until, step, v_0, v_th, a_v, b_g, c_vg, fired
):
    for a in prange(n_active):
        _update(active[a], a, v, g, refractory_until, step, v_0, v_th, a_v, b_g, c_vg, fired)


@njit(cache=True)
def _run(
    steps, t0, indptr, indices, weight, v, g, refractory_until, refractory_steps,
    active_flag, active, n_active, spike_ring, ring_count, stim_index, stim_prob, counts,
    fired, v_0, v_rst, v_th, a_v, b_g, c_vg, delay, w_syn, kick, tol,
):  # fmt: skip
    # Operation order within a step matches Brian2's default schedule:
    # refractoriness, state update, thresholds, synaptic and Poisson effects, resets.
    ring = spike_ring.shape[0]
    for step in range(t0, t0 + steps):
        # 1. Integrate non-refractory active neurons exactly and test the threshold.
        if n_active >= PARALLEL_MIN_ACTIVE:
            _integrate_parallel(active, n_active, v, g, refractory_until, step, v_0, v_th,
                                a_v, b_g, c_vg, fired)  # fmt: skip
        else:
            _integrate(active, n_active, v, g, refractory_until, step, v_0, v_th,
                       a_v, b_g, c_vg, fired)  # fmt: skip
        # 2. Spiking neurons become refractory now: like Brian2's conditional writes on
        #    `(unless refractory)` variables, inputs to refractory neurons are dropped.
        out_slot = (step + delay) % ring
        for a in range(n_active):
            if fired[a]:
                i = active[a]
                refractory_until[i] = step + 1 + refractory_steps[i]
                counts[i] += 1
                spike_ring[out_slot, ring_count[out_slot]] = i
                ring_count[out_slot] += 1
        integrated = n_active
        # 3. Deliver spikes emitted `delay` steps ago.
        slot = step % ring
        for s in range(ring_count[slot]):
            source = spike_ring[slot, s]
            for e in range(indptr[source], indptr[source + 1]):
                target = indices[e]
                if step >= refractory_until[target]:
                    g[target] += weight[e] * w_syn
                    if not active_flag[target]:
                        active_flag[target] = True
                        active[n_active] = target
                        n_active += 1
        ring_count[slot] = 0
        # 4. Poisson sensory drive.
        for k in range(stim_index.shape[0]):
            target = stim_index[k]
            if (
                stim_prob[k] > 0.0
                and np.random.random() < stim_prob[k]
                and step >= refractory_until[target]
            ):
                v[target] += kick
                if not active_flag[target]:
                    active_flag[target] = True
                    active[n_active] = target
                    n_active += 1
        # 5. Reset neurons that fired and retire neurons back at rest.
        kept = 0
        for a in range(n_active):
            i = active[a]
            if a < integrated and fired[a]:
                v[i] = v_rst
                g[i] = 0.0
            if step + 1 >= refractory_until[i] and abs(v[i] - v_0) < tol and abs(g[i]) < tol:
                v[i] = v_0
                g[i] = 0.0
                active_flag[i] = False
            else:
                active[kept] = i
                kept += 1
        n_active = kept
        # Keep the active list in index order: sequential memory access is several
        # times faster than the arbitrary order in which neurons were woken up.
        if step % REORDER_EVERY_STEPS == 0 and n_active > 1024:
            n_active = 0
            for i in range(active_flag.shape[0]):
                if active_flag[i]:
                    active[n_active] = i
                    n_active += 1
    return n_active


def _kernel_array(values, dtype):
    flags = ["C_CONTIGUOUS", "WRITEABLE", "OWNDATA"]
    return np.require(np.asarray(values, dtype=dtype), dtype, flags)


class LIFSimulator:
    def __init__(self, connectome, params=None, seed=0):
        self.connectome = connectome
        self.params = params or LIFParams.matched_to(connectome.manifest)
        self.seed = seed
        p, n = self.params, connectome.size
        # One array type per argument keeps numba to a single compiled specialization.
        # Several cached specializations of a kernel that calls a parallel function
        # crashed on load (read-only memmaps and writable arrays type differently).
        self.indptr = _kernel_array(connectome.indptr, np.int64)
        self.indices = _kernel_array(connectome.indices, np.int32)
        self.weight = _kernel_array(connectome.weight, np.float32)
        self.delay = max(1, round(p.t_dly_ms / p.dt_ms))
        # Brian2 keeps a neuron refractory while t < t_spike + t_rfc.
        self.default_refractory = max(0, math.ceil(p.t_rfc_ms / p.dt_ms - 1e-9) - 1)
        self.a_v = math.exp(-p.dt_ms / p.t_mbr_ms)
        self.b_g = math.exp(-p.dt_ms / p.tau_ms)
        self.c_vg = p.tau_ms / (p.tau_ms - p.t_mbr_ms) * (self.b_g - self.a_v)
        self.v = np.empty(n, np.float64)
        self.g = np.empty(n, np.float64)
        self.refractory_until = np.empty(n, np.int64)
        self.refractory_steps = np.full(n, self.default_refractory, np.int64)
        self.active_flag = np.empty(n, np.bool_)
        self.active = np.empty(n, np.int32)
        self.fired = np.zeros(n, np.bool_)
        self.spike_ring = np.empty((self.delay + 1, n), np.int32)
        self.ring_count = np.empty(self.delay + 1, np.int64)
        self.counts = np.zeros(n, np.int32)
        self.stim_index = np.zeros(0, np.int32)
        self.stim_prob = np.zeros(0, np.float64)
        self.reset()

    def reset(self):
        self.v.fill(self.params.v_0)
        self.g.fill(0)
        self.refractory_until.fill(0)
        self.active_flag.fill(False)
        self.ring_count.fill(0)
        self.counts.fill(0)
        self.n_active = 0
        self.step_index = 0
        _seed(self.seed)

    @property
    def time_ms(self):
        return self.step_index * self.params.dt_ms

    def set_stimulus(self, drive):
        """``drive`` is an iterable of ``(neuron_indices, poisson_rate_hz)`` pairs.

        Driven neurons get no refractory period (as in the reference model) until
        they are left out of a later stimulus.
        """
        index, prob = [], []
        for neurons, rate in drive:
            if not math.isfinite(rate) or rate < 0:
                raise ValueError("Stimulus rates must be finite and non-negative")
            neurons = np.asarray(neurons, np.int32).ravel()
            index.append(neurons)
            prob.append(np.full(len(neurons), min(rate * self.params.dt_ms / 1000, 1.0)))
        self.refractory_steps[self.stim_index] = self.default_refractory
        self.stim_index = _kernel_array(np.concatenate(index) if index else [], np.int32)
        self.stim_prob = _kernel_array(np.concatenate(prob) if prob else [], np.float64)
        self.refractory_steps[self.stim_index] = 0

    def run(self, duration_ms):
        steps = round(duration_ms / self.params.dt_ms)
        if steps < 0:
            raise ValueError("duration must be non-negative")
        p = self.params
        self.n_active = int(_run(
            int(steps), int(self.step_index), self.indptr, self.indices, self.weight, self.v,
            self.g, self.refractory_until, self.refractory_steps, self.active_flag, self.active,
            int(self.n_active), self.spike_ring, self.ring_count, self.stim_index, self.stim_prob,
            self.counts, self.fired, float(p.v_0), float(p.v_rst), float(p.v_th), float(self.a_v),
            float(self.b_g), float(self.c_vg), int(self.delay), float(p.w_syn),
            float(p.poisson_kick_mv), float(p.rest_tolerance),
        ))  # fmt: skip
        self.step_index += steps
        return steps

    def take_counts(self):
        """Spike counts per neuron since the previous call."""
        counts = self.counts.copy()
        self.counts.fill(0)
        return counts
