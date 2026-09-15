"""Spike-for-spike comparison of the numba LIF kernel with Brian2.

Brian2 is the simulator of the reference model (Shiu et al. 2024). Both engines get
the same random network and the same deterministic input kicks; every spike
(neuron, time step) must match. Requires the optional ``brian2`` package.
"""

import logging

import numpy as np

from flybrain_robot.brain.lif import LIFParams, LIFSimulator
from flybrain_robot.connectome.synthetic import random_connectome

STEPS = 3000


def kick_schedule(n, seed, driven_count=40, events=900):
    rng = np.random.default_rng(seed)
    driven = rng.choice(n, driven_count, replace=False)
    pairs = zip(rng.integers(0, STEPS, events), rng.choice(driven, events))
    return driven, sorted({(int(s), int(i)) for s, i in pairs})


def run_brian2(graph, params, driven, events):
    from brian2 import (
        Network,
        NeuronGroup,
        SpikeGeneratorGroup,
        SpikeMonitor,
        Synapses,
        defaultclock,
        ms,
        mV,
        prefs,
    )

    logging.getLogger("brian2").setLevel(logging.ERROR)
    prefs.codegen.target = "numpy"
    defaultclock.dt = params.dt_ms * ms
    namespace = dict(
        v_0=params.v_0 * mV,
        v_rst=params.v_rst * mV,
        v_th=params.v_th * mV,
        t_mbr=params.t_mbr_ms * ms,
        tau=params.tau_ms * ms,
    )
    # Equations, reset and refractoriness exactly as in Shiu et al.'s model.py.
    neurons = NeuronGroup(
        graph.size,
        """dv/dt = (v_0 - v + g) / t_mbr : volt (unless refractory)
           dg/dt = -g / tau : volt (unless refractory)
           rfc : second""",
        method="linear",
        threshold="v > v_th",
        reset="v = v_rst; g = 0 * mV",
        refractory="rfc",
        namespace=namespace,
    )
    neurons.v = params.v_0 * mV
    neurons.rfc = params.t_rfc_ms * ms
    neurons.rfc[driven] = 0 * ms
    pre = np.repeat(np.arange(graph.size), np.diff(graph.indptr))
    synapses = Synapses(neurons, neurons, "w : volt", on_pre="g += w", delay=params.t_dly_ms * ms)
    synapses.connect(i=pre, j=np.asarray(graph.indices))
    synapses.w = np.asarray(graph.weight) * params.w_syn * mV
    steps, targets = zip(*events)
    generator = SpikeGeneratorGroup(
        graph.size, np.array(targets), np.array(steps) * params.dt_ms * ms
    )
    kicks = Synapses(generator, neurons, on_pre=f"v += {params.poisson_kick_mv} * mV")
    kicks.connect(j="i")
    monitor = SpikeMonitor(neurons)
    Network(neurons, synapses, generator, kicks, monitor).run(STEPS * params.dt_ms * ms)
    times = np.round(np.asarray(monitor.t / ms) / params.dt_ms).astype(int)
    return sorted(zip(times.tolist(), np.asarray(monitor.i).tolist()))


def run_numba(graph, params, driven, events):
    sim = LIFSimulator(graph, params=params)
    by_step = {}
    for step, neuron in events:
        by_step.setdefault(step, []).append(neuron)
    driven_set = set(driven.tolist())
    spikes = []
    for step in range(STEPS):
        now = by_step.get(step, [])
        rest = sorted(driven_set - set(now))
        # A rate of 1/dt gives a kick with probability 1 in this step.
        sim.set_stimulus([(now, 1000 / params.dt_ms), (rest, 0.0)])
        before = sim.counts.copy()
        sim.run(params.dt_ms)
        spikes += [(step, int(i)) for i in np.flatnonzero(sim.counts != before)]
    return spikes


def compare(seed, n=400, density=0.03, w_syn=5.0):
    graph = random_connectome(n=n, density=density, seed=seed)
    params = LIFParams(w_syn=w_syn)
    driven, events = kick_schedule(graph.size, seed)
    reference = run_brian2(graph, params, driven, events)
    ours = run_numba(graph, params, driven, events)
    mismatch = next((k for k, (a, b) in enumerate(zip(reference, ours)) if a != b), None)
    if mismatch is None and len(reference) != len(ours):
        mismatch = min(len(reference), len(ours))
    return {
        "seed": seed,
        "neurons": n,
        "edges": int(len(graph.indices)),
        "steps": STEPS,
        "brian2_spikes": len(reference),
        "numba_spikes": len(ours),
        "spiking_neurons": len({i for _, i in reference}),
        "identical": reference == ours,
        "first_mismatch_index": mismatch,
    }
