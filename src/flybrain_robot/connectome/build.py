"""Download the public MaleCNS v1.0 tables and compile them into a simulation graph.

Usage:
    python -m flybrain_robot.connectome.build download --data data/malecns-v1.0
    python -m flybrain_robot.connectome.build compile --data data/malecns-v1.0
    python -m flybrain_robot.connectome.build info --data data/malecns-v1.0

Requires the ``connectome`` extra (pandas, pyarrow). The dataset is published by
HHMI Janelia under CC BY 4.0 and is never bundled with this repository.
"""

import argparse
import hashlib
import json
import shutil
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np

from .graph import FORMAT_VERSION, Connectome, load_connectome, save_connectome

BASE_URL = (
    "https://storage.googleapis.com/flyem-male-cns/v1.0/connectome-data/flat-connectome/"
)
LICENSE = "CC BY 4.0, https://male-cns.janelia.org/download/"
ANNOTATIONS = "body-annotations-male-cns-v1.0-minconf-0.5.feather"
TRANSMITTERS = "body-neurotransmitters-male-cns-v1.0.feather"
WEIGHTS = "connectome-weights-male-cns-v1.0-minconf-0.5.feather"
# Pinned on 2026-09-15 from the Janelia bucket. A mismatch means the upstream
# release changed; results would no longer be comparable to docs/VALIDATION.md.
FILES = {
    ANNOTATIONS: (14483314, "2177e246113e4cfbf1e7772ec37c6da1955ff22e8063d0b1f833101f99a9a3b2"),
    TRANSMITTERS: (43282834, "95c9289220663abeb3409f3ad9e5a7f8a53f8093f5139d15502cd08da8879621"),
    WEIGHTS: (1051241946, "e35da783d1c686b2b58b3b87cd6a403ae43bfcfba8bff28e08ef752c1a56afc1"),
}
# Sign convention of Shiu et al. 2024 (GABA and glutamate inhibit), extended with
# histamine, which acts on histamine-gated chloride channels in Drosophila.
INHIBITORY = {"gaba", "glutamate", "histamine"}


def sha256(path, chunk=1 << 22):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        while block := file.read(chunk):
            digest.update(block)
    return digest.hexdigest()


def download(data_dir, verify=True):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, (size, digest) in FILES.items():
        target = data_dir / name
        if not (target.exists() and target.stat().st_size == size):
            print(f"Downloading {name} ({size / 1e6:.0f} MB)")
            partial = target.with_suffix(".partial")
            with urllib.request.urlopen(BASE_URL + name) as response, open(partial, "wb") as out:
                shutil.copyfileobj(response, out, 1 << 22)
            partial.replace(target)
        if verify:
            actual = sha256(target)
            if actual != digest:
                raise ValueError(f"SHA-256 mismatch for {name}: {actual}")
            print(f"Verified {name}")


def neuron_transmitters(transmitters, body_ids):
    """Per-neuron transmitter: body consensus, then cell-type prediction, else unclear."""
    table = transmitters.set_index("body").reindex(body_ids)
    nt = table["consensus_nt"].where(table["consensus_nt"] != "unclear")
    by_type = table["celltype_predicted_nt"]
    nt = nt.fillna(by_type.where(by_type != "unclear"))
    return nt.fillna("unclear").to_numpy(dtype=str)


def neuron_sides(neurons):
    """Soma side; for sensory neurons (soma outside the CNS) the side their root enters.

    ``somaSide`` is empty for 18,003 neurons of the release, all sensory; 17,867 of
    them have ``rootSide``. ``rootSide`` "unknown" counts as no side. The instance
    suffix (``_L``, ``_R``, ``_M``) is the last fallback.
    """
    side = neurons["somaSide"].where(neurons["somaSide"].isin(["L", "R", "M"]))
    side = side.fillna(neurons["rootSide"].where(neurons["rootSide"].isin(["L", "R", "M"])))
    suffix = neurons["instance"].fillna("").str.extract(r"_([LRM])$", expand=False)
    return side.fillna(suffix).fillna("").to_numpy(dtype=str)


def hex_column(neurons, name):
    """Optic-lobe column coordinate as int16, -1 where the neuron has no column."""
    return neurons[name].fillna(-1).to_numpy().astype(np.int16)


def peak_memory_gb():
    """Peak memory of this process, so the README figure is measured, not remembered.

    ``None`` where it cannot be read: psutil is optional, and resource.getrusage is
    absent on Windows.
    """
    try:
        import psutil

        info = psutil.Process().memory_info()
        peak = getattr(info, "peak_wset", None) or info.rss
        return round(peak / 2**30, 2)
    except ImportError:
        pass
    try:
        import resource

        kilobytes = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return round(kilobytes / 2**20, 2)  # Linux reports kB, macOS bytes
    except (ImportError, AttributeError):
        return None


def compile_graph(data_dir, verify=True):
    import pandas as pd
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.feather as feather

    data_dir = Path(data_dir)
    if verify:
        for name, (_, digest) in FILES.items():
            if sha256(data_dir / name) != digest:
                raise ValueError(f"{name} does not match the pinned MaleCNS v1.0 release")
    started = time.monotonic()
    annotations = pd.read_feather(data_dir / ANNOTATIONS)
    # The 166,700 bodies with a curated superclass are the neurons of the release.
    neurons = annotations[annotations["superclass"].notna()].sort_values("bodyId")
    body_ids = neurons["bodyId"].to_numpy(np.int64)
    nt = neuron_transmitters(pd.read_feather(data_dir / TRANSMITTERS), body_ids)
    sign = np.where(np.isin(nt, list(INHIBITORY)), -1, 1).astype(np.int8)

    edges = feather.read_table(data_dir / WEIGHTS)
    total_edges = edges.num_rows
    ids = pa.array(body_ids)
    keep = pc.and_(
        pc.is_in(edges["body_pre"], value_set=ids), pc.is_in(edges["body_post"], value_set=ids)
    )
    edges = edges.filter(keep)
    pre = np.searchsorted(body_ids, edges["body_pre"].to_numpy()).astype(np.int32)
    post = np.searchsorted(body_ids, edges["body_post"].to_numpy()).astype(np.int32)
    count = edges["weight"].to_numpy().astype(np.int32)
    del edges
    order = np.lexsort((post, pre))
    pre, post, count = pre[order], post[order], count[order]
    indptr = np.zeros(len(body_ids) + 1, np.int64)
    np.cumsum(np.bincount(pre, minlength=len(body_ids)), out=indptr[1:])

    def text(column):
        return neurons[column].fillna("").to_numpy(dtype=str)

    manifest = {
        "format_version": FORMAT_VERSION,
        "dataset": "MaleCNS v1.0 (minconf 0.5)",
        "license": LICENSE,
        "attribution": "HHMI Janelia FlyEM, Google Research, MRC LMB and collaborators",
        "sources": {n: {"bytes": size, "sha256": digest} for n, (size, digest) in FILES.items()}
        if verify
        else "not verified (--no-verify)",
        "neurons": int(len(body_ids)),
        "edges_in_release": int(total_edges),
        "neuron_to_neuron_edges": int(len(post)),
        "synapses": int(count.sum()),
        "inhibitory_neurons": int((sign < 0).sum()),
        "transmitter_counts": {k: int(v) for k, v in zip(*np.unique(nt, return_counts=True))},
        "sign_rule": "gaba, glutamate, histamine inhibitory; all other or unclear excitatory",
        "side_rule": "somaSide, else rootSide (sensory neurons), else instance suffix",
        "neurons_with_optic_column": int(neurons["assignedOlHex1"].notna().sum()),
        "compile_seconds": round(time.monotonic() - started, 1),
        "compile_peak_memory_gb": peak_memory_gb(),
    }
    graph = Connectome(
        body_id=body_ids,
        type=text("type"),
        instance=text("instance"),
        side=neuron_sides(neurons),
        hex1=hex_column(neurons, "assignedOlHex1"),
        hex2=hex_column(neurons, "assignedOlHex2"),
        superclass=text("superclass"),
        transmitter=nt,
        sign=sign,
        indptr=indptr,
        indices=post,
        weight=count * sign[pre],
        manifest=manifest,
    )
    save_connectome(graph, data_dir)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=["download", "compile", "info"])
    parser.add_argument("--data", default="data/malecns-v1.0", help="Dataset directory")
    parser.add_argument("--no-verify", action="store_true", help="Skip SHA-256 checks")
    args = parser.parse_args(argv)
    try:
        if args.command == "download":
            download(args.data, verify=not args.no_verify)
        elif args.command == "compile":
            print(json.dumps(compile_graph(args.data, verify=not args.no_verify), indent=2))
        else:
            graph = load_connectome(args.data)
            print(json.dumps(graph.manifest, indent=2))
    except (OSError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
