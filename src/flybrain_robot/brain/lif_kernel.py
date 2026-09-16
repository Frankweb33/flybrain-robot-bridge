"""Per-neuron LIF arithmetic shared by the CPU (numba.njit) and GPU (numba.cuda) kernels.

This module holds plain Python source and imports no numba: ``lif.py`` compiles
``neuron_block`` with ``njit`` and ``lif_cuda.py`` with ``cuda.jit(device=True)``, so
both kernels execute the same operations in the same order.

Why whole blocks: a spike emitted at step t reaches its targets at t + delay. Within a
block of ``delay`` steps every input a neuron receives therefore comes from spikes of
the previous block, which are known before the block starts. Neurons do not interact
inside a block and can be advanced independently, in parallel, for all its steps.

Why fixed point: the synaptic input of a step is summed as int64 (weights scaled by
2**FIXED_SHIFT) and added to g once. Integer sums do not depend on the order of the
spikes, so parallel CPU threads and GPU threads give bit-identical results.

Why a hash instead of a random stream: the Poisson draw of neuron i at step t is
splitmix64(seed, t, i). It needs no state and is the same on every device.

Why an explicit fused multiply-add: NVIDIA's code generator contracts ``a * b + c``
into one FMA instruction by default, which rounds differently in about a quarter of
cases. The membrane update therefore calls ``fma`` explicitly, a correctly rounded
IEEE 754 operation on both devices, and leaves nothing else to contract. ``fma`` is a
module global that ``specialize`` replaces with the CPU or GPU implementation.
"""

import math
import types

import numpy as np

FIXED_SHIFT = 20
MAX_BLOCK_STEPS = 32  # spike bits of a block are kept in a uint32

fma = math.fma  # replaced per device by ``specialize``


def specialize(function, **names):
    """Same bytecode as ``function`` with some global names bound differently."""
    return types.FunctionType(
        function.__code__, {**function.__globals__, **names}, function.__name__,
        function.__defaults__, function.__closure__,
    )  # fmt: skip


def neuron_block(
    i, s0, s1, block_start, seed, v, g, h, refractory_until, refractory_steps, dg, dh, bits,
    counts, prob, v_rest, v_rst, v_th, a_v, b_g, b_h, c_vg, c_vh, kick, inv_scale, tol,
    block_len,
):  # fmt: skip
    """Advance neuron ``i`` through block steps [s0, s1).

    Operation order per step matches Brian2's schedule: state update, threshold,
    synaptic and Poisson input (dropped while refractory), reset. ``v_rest`` holds the
    resting potential of each neuron: above threshold it fires tonically on its own,
    which is how a graded, tonically releasing cell is represented here.

    ``g`` collects input from synapses with the reference time constant and ``h`` from
    those of presynaptic cells marked slow; with one time constant both pools behave
    identically and ``h`` stays empty.
    """
    v_0 = v_rest[i]
    vi = v[i]
    gi = g[i]
    hi = h[i]
    ru = refractory_until[i]
    p = prob[i]
    row = i * block_len
    for s in range(s0, s1):
        step = block_start + s
        fired = False
        if step >= ru:
            # v_0 + (v - v_0) a_v + g c_vg + h c_vh
            vi = fma(hi, c_vh, fma(gi, c_vg, fma(vi - v_0, a_v, v_0)))
            gi = gi * b_g
            hi = hi * b_h
            if vi > v_th:
                fired = True
                ru = step + 1 + refractory_steps[i]
                counts[i] += 1
                bits[i] = bits[i] | (np.uint32(1) << np.uint32(s))
        if step >= ru:
            d = dg[row + s]
            if d != 0:
                gi += d * inv_scale
            d = dh[row + s]
            if d != 0:
                hi += d * inv_scale
            if p > 0.0:
                x = (
                    (np.uint64(seed) * np.uint64(0x9E3779B97F4A7C15))
                    ^ (np.uint64(step) * np.uint64(0xBF58476D1CE4E5B9))
                    ^ (np.uint64(i) * np.uint64(0x94D049BB133111EB))
                )
                x = x + np.uint64(0x9E3779B97F4A7C15)
                x = (x ^ (x >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
                x = (x ^ (x >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
                x = x ^ (x >> np.uint64(31))
                if (x >> np.uint64(11)) * 1.1102230246251565e-16 < p:  # 2**-53
                    vi += kick
        if fired:
            vi = v_rst
            gi = 0.0
            hi = 0.0
        if step + 1 >= ru and abs(vi - v_0) < tol and abs(gi) < tol and abs(hi) < tol:
            vi = v_0
            gi = 0.0
            hi = 0.0
    v[i] = vi
    g[i] = gi
    h[i] = hi
    refractory_until[i] = ru
