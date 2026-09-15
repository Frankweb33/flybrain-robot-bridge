"""Optic-lobe columns for neurons that the release does not assign one, from the wiring.

MaleCNS gives ``assignedOlHex1/2`` to 23,720 neurons, but not to T4, T5, Tm3, L4 or the
photoreceptors, and not to the two CT1 cells. A columnar neuron's partners sit in and
around its own column, so the synapse-weighted centroid of the partner columns recovers
it: on the neurons that do have a column, and using only their partners' columns, the
centroid is exact for 97.7% of them and within one column for 99.3% (measured by
:func:`accuracy` on every run, reported in VALIDATION.md).

Wide-field cells have no single column, and the same measurement separates them: the
synapse-weighted spread of their partner columns is 2.1 for LPLC2, 3.0 for LC4 and 14.0
for CT1, against 0.8-1.3 for the columnar cells that lack a coordinate. Neurons above
``MAX_SPREAD`` keep no column.

CT1 is a special case that :func:`compartmentalize` handles instead: one cell per eye
contacts every column, but its compartments are electrically isolated from each other
(Meier & Borst 2019), so as a single point neuron it would inhibit the whole eye at once.
"""

import numpy as np

from .graph import Connectome

MIN_SYNAPSES = 20  # below this a centroid is guesswork
MAX_SPREAD = 1.5  # columns; above this the cell is wide-field, not columnar
ROUNDS = 2  # a second pass reaches cells whose partners were imputed in the first


def hex_distance(q1, r1, q2, r2):
    """Distance on the hexagonal lattice in axial coordinates."""
    dq, dr = q1 - q2, r1 - r2
    return (np.abs(dq) + np.abs(dr) + np.abs(dq + dr)) / 2


def _partners(graph):
    """Undirected (neuron, partner, synapses) arrays; a reciprocal pair counts twice."""
    indptr, indices = np.asarray(graph.indptr), np.asarray(graph.indices)
    weight = np.abs(np.asarray(graph.weight)).astype(np.float64)
    pre = np.repeat(np.arange(graph.size, dtype=np.int32), np.diff(indptr))
    return (np.concatenate([pre, indices]), np.concatenate([indices, pre]),
            np.concatenate([weight, weight]))  # fmt: skip


def estimate(graph, hex1, hex2, partners=None):
    """Synapse-weighted centroid of the partner columns, and the spread around it.

    Returns ``(hex1, hex2, synapses, spread)``; ``synapses`` counts only the synapses
    with partners that have a column, and a neuron's own column is never used.
    """
    neuron, partner, weight = partners if partners is not None else _partners(graph)
    use = hex1[partner] >= 0
    neuron, partner, weight = neuron[use], partner[use], weight[use]
    size = graph.size
    total = np.bincount(neuron, weights=weight, minlength=size)
    safe = np.maximum(total, 1e-9)
    centre1 = np.bincount(neuron, weights=weight * hex1[partner], minlength=size) / safe
    centre2 = np.bincount(neuron, weights=weight * hex2[partner], minlength=size) / safe
    distance = hex_distance(hex1[partner], hex2[partner], centre1[neuron], centre2[neuron])
    spread = np.bincount(neuron, weights=weight * distance, minlength=size) / safe
    return np.rint(centre1).astype(np.int32), np.rint(centre2).astype(np.int32), total, spread


def accuracy(graph):
    """How often the rule recovers the column of the neurons that already have one."""
    hex1, hex2 = np.asarray(graph.hex1).astype(np.int32), np.asarray(graph.hex2).astype(np.int32)
    guess1, guess2, total, _ = estimate(graph, hex1, hex2)
    checked = (hex1 >= 0) & (total >= MIN_SYNAPSES)
    error = hex_distance(guess1[checked], guess2[checked], hex1[checked], hex2[checked])
    return dict(neurons=int(checked.sum()), exact=round(float((error == 0).mean()), 4),
                within_one=round(float((error <= 1).mean()), 4),
                median_error=float(np.median(error)))  # fmt: skip


def impute_columns(graph, rounds=ROUNDS, min_synapses=MIN_SYNAPSES, max_spread=MAX_SPREAD):
    """Copy of the graph in which columnar neurons without a column get one."""
    hex1 = np.array(graph.hex1, np.int32)
    hex2 = np.array(graph.hex2, np.int32)
    partners = _partners(graph)
    added = []
    for _ in range(rounds):
        guess1, guess2, total, spread = estimate(graph, hex1, hex2, partners)
        fill = (hex1 < 0) & (total >= min_synapses) & (spread <= max_spread)
        if not fill.any():
            break
        hex1[fill], hex2[fill] = guess1[fill], guess2[fill]
        added.append(int(fill.sum()))
    manifest = dict(graph.manifest, imputed_columns=dict(
        rule=f"synapse-weighted partner centroid, >= {min_synapses} synapses, "
             f"spread <= {max_spread} columns, {rounds} rounds",
        added_per_round=added, neurons_with_column=int((hex1 >= 0).sum()),
        accuracy=accuracy(graph),
    ))  # fmt: skip
    return Connectome(**{**graph.__dict__, "hex1": hex1.astype(np.int16),
                         "hex2": hex2.astype(np.int16), "manifest": manifest})  # fmt: skip


def compartmentalize(graph, types=("CT1",)):
    """Copy in which each cell of ``types`` becomes one neuron per partner column.

    Every synapse keeps its partner, its count and its sign; only the cell body it
    belongs to is split, so a compartment can no longer drive the whole eye. Synapses
    with partners that have no column share one remaining compartment per cell.
    """
    split = np.isin(np.asarray(graph.type), list(types))
    if not split.any():
        raise ValueError(f"No neurons of type {types}")
    size = graph.size
    indptr, indices = np.asarray(graph.indptr), np.asarray(graph.indices)
    pre = np.repeat(np.arange(size, dtype=np.int64), np.diff(indptr))
    post = indices.astype(np.int64)
    hex1, hex2 = np.asarray(graph.hex1).astype(np.int64), np.asarray(graph.hex2).astype(np.int64)
    column = np.where(hex1 >= 0, hex1 * 128 + hex2, -1)  # one integer per column

    # Every synapse of a split cell is labelled with the column of its partner; an edge
    # between two split cells is labelled once for each end.
    out_side, in_side = split[pre], split[post]
    cells = np.concatenate([pre[out_side], post[in_side]])
    keys = np.concatenate([column[post[out_side]], column[pre[in_side]]])
    unique, inverse = np.unique(np.stack([cells, keys], axis=1), axis=0, return_inverse=True)
    parent, new_column = unique[:, 0], unique[:, 1]
    new_index = np.empty(len(unique), np.int64)
    _, first = np.unique(parent, return_index=True)  # the first compartment keeps the parent row
    extra = np.ones(len(unique), bool)
    extra[first] = False
    new_index[first] = parent[first]
    grow = int(extra.sum())
    new_index[extra] = size + np.arange(grow)

    inverse = np.ravel(inverse)
    pre[out_side] = new_index[inverse[: int(out_side.sum())]]
    post[in_side] = new_index[inverse[int(out_side.sum()) :]]
    fields = {}
    for name in ("body_id", "type", "instance", "side", "superclass", "transmitter", "sign"):
        values = np.asarray(getattr(graph, name))
        fields[name] = np.concatenate([values, values[parent[extra]]])
    for name, source in (("hex1", 0), ("hex2", 1)):
        values = np.array(getattr(graph, name), np.int16)
        column_value = np.where(new_column >= 0,
                                new_column // 128 if source == 0 else new_column % 128, -1)
        values = np.concatenate([values, np.zeros(grow, np.int16)])
        values[new_index] = column_value.astype(np.int16)
        fields[name] = values
    fields["instance"] = fields["instance"].astype(object)
    for k, index in enumerate(new_index):
        suffix = f"_col{new_column[k] // 128}_{new_column[k] % 128}" if new_column[k] >= 0 else "_rest"
        fields["instance"][index] = str(fields["instance"][index]).split("_col")[0] + suffix
    fields["instance"] = fields["instance"].astype(str)

    order = np.lexsort((post, pre))
    pre, post = pre[order], post[order]
    weight = np.asarray(graph.weight)[order]
    new_size = size + grow
    new_indptr = np.zeros(new_size + 1, np.int64)
    np.cumsum(np.bincount(pre, minlength=new_size), out=new_indptr[1:])
    manifest = dict(graph.manifest, compartments=dict(
        types=list(types), cells=int(split.sum()), compartments=len(unique),
        edges_without_partner_column=int((keys < 0).sum()),
    ))  # fmt: skip
    return Connectome(indptr=new_indptr, indices=post.astype(np.int32), weight=weight,
                      manifest=manifest, **fields)  # fmt: skip
