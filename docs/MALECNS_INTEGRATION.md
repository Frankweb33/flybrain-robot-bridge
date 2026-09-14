# MaleCNS integration target

The current backend deliberately raises `Not implemented yet` after validating
that a configured local dataset path exists. It does not download or parse data.

Obtain dataset access and download information from the
[Janelia Male CNS page](https://www.janelia.org/project-team/flyem/male-cns-connectome).
Follow the source's data license and attribution requirements independently of
this repository's MIT license.

A future implementation needs:

1. A versioned graph loader retaining neuron IDs, weights and annotations.
2. Explicit sensory and motor population mappings, reviewed against the science.
3. A documented neuron/synapse dynamics model and time-unit convention.
4. Reproducible small-subgraph tests before scaling up.

`load_graph`, `feed_sensory`, `step`, `get_motor_activity` and `reset` are the
extension seams. Connectivity alone does not define a validated dynamical model.
Do not return mock activity under the MaleCNS name.
