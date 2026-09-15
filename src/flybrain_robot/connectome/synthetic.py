"""Small synthetic connectomes in the compiled format, for tests and engine checks without data."""

import numpy as np

from flybrain_robot.connectome.graph import FORMAT_VERSION, Connectome


def connectome_from_edges(n, pre, post, weight, types=None, sides=None, hex1=None, hex2=None):
    pre, post, weight = (np.asarray(x) for x in (pre, post, weight))
    order = np.lexsort((post, pre))
    indptr = np.zeros(n + 1, np.int64)
    np.cumsum(np.bincount(pre, minlength=n), out=indptr[1:])
    text = np.array([""] * n)
    no_column = np.full(n, -1, np.int16)
    types = np.asarray(types) if types is not None else np.array([f"T{i}" for i in range(n)])
    return Connectome(
        body_id=np.arange(n, dtype=np.int64) + 10000,
        type=types,
        instance=text,
        side=np.asarray(sides) if sides is not None else text,
        hex1=np.asarray(hex1, np.int16) if hex1 is not None else no_column,
        hex2=np.asarray(hex2, np.int16) if hex2 is not None else no_column,
        superclass=text,
        transmitter=np.array(["acetylcholine"] * n),
        sign=np.ones(n, np.int8),
        indptr=indptr,
        indices=post[order].astype(np.int32),
        weight=weight[order].astype(np.int32),
        manifest={
            "format_version": FORMAT_VERSION,
            "neurons": n,
            "synapses": int(np.abs(weight).sum()),
        },
    )


def random_connectome(n=400, density=0.03, seed=0, inhibitory_fraction=0.3):
    rng = np.random.default_rng(seed)
    mask = rng.random((n, n)) < density
    np.fill_diagonal(mask, False)
    pre, post = np.nonzero(mask)
    sign = np.where(rng.random(n) < inhibitory_fraction, -1, 1)
    weight = rng.geometric(0.3, len(pre)) * sign[pre]
    return connectome_from_edges(n, pre, post, weight)
