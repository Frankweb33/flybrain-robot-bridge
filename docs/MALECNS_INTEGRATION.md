# MaleCNS integration

`--backend malecns` runs a spiking simulation of every neuron in the
[MaleCNS v1.0 connectome](https://male-cns.janelia.org/) (HHMI Janelia FlyEM,
Google Research, MRC LMB and collaborators): brain and ventral nerve cord of an
adult male *Drosophila melanogaster*. Measured results are in
[VALIDATION.md](VALIDATION.md); this page explains what is simulated and which
choices were made by hand.

## Data

`python -m flybrain_robot.connectome.build download` fetches three public tables
from the Janelia bucket and checks their SHA-256 against the values pinned in
`connectome/build.py`:

| file | used for |
|---|---|
| `body-annotations-male-cns-v1.0-minconf-0.5.feather` | neuron list, cell type, soma side |
| `body-neurotransmitters-male-cns-v1.0.feather` | predicted transmitter → synapse sign |
| `connectome-weights-male-cns-v1.0-minconf-0.5.feather` | synapse counts between bodies |

`compile` keeps the 166,700 bodies that carry a curated `superclass` (the neurons
of the release) and every connection between them, and stores a CSR graph of
signed synapse counts in `data/malecns-v1.0/compiled/`. The data is CC BY 4.0 and
is never committed to this repository; attribute the MaleCNS authors when you
publish results. CC BY does not imply that they endorse this project.

Transmitter per neuron: the body's consensus prediction, else the prediction for
its cell type, else `unclear`. GABA, glutamate and histamine are inhibitory;
everything else, including `unclear`, is excitatory.

Side per neuron: the soma side; sensory neurons have their soma outside the CNS, so
for them the side their root enters (`rootSide`), else the `_L`/`_R` suffix of the
instance name. 548 neurons remain without a side. Optic-lobe column coordinates
(`assignedOlHex1/2`) are kept for the 23,720 neurons that have them.

## Neuron model

Leaky integrate-and-fire with conductance-like exponential synapses, using the
equations and parameters of Shiu et al. 2024, *A Drosophila computational brain
model reveals sensorimotor processing*, Nature 634, 210–219
([code](https://github.com/philshiu/Drosophila_brain_model)): resting and reset
potential −52 mV, threshold −45 mV, membrane time constant 20 ms, synaptic time
constant 5 ms, refractory period 2.2 ms, synaptic delay 1.8 ms, time step 0.1 ms,
Poisson input at up to 150 Hz.

`tests/test_lif_reference.py` runs the same networks in Brian2, the simulator of the
reference model, and requires every spike to match.

### Kernels

There is a CPU kernel (`brain/lif.py`, numba) and a GPU kernel
(`brain/lif_cuda.py`, numba-cuda-mlir). `device: auto` uses the GPU when one is
available. Both compile the same per-neuron update (`brain/lif_kernel.py`) and are
built so that they give the same result to the last bit:

- **Blocks.** A spike reaches its targets 1.8 ms (18 steps) later, so within an
  18-step block every input is already known and neurons do not interact. Each block
  is one parallel pass over neurons, then one delivery of the block's spikes.
- **Fixed-point input.** The synaptic input of a step is summed as integers
  (weights × w_syn × 2²⁰, rounded) and added to g once, so the order of spikes, and
  of GPU threads, does not matter. The rounding changes a weight by at most
  5 × 10⁻⁷ mV.
- **Counter-based random numbers.** The Poisson draw of neuron i at step t is a
  splitmix64 hash of (seed, t, i), identical on every device.
- **Explicit fused multiply-add.** NVIDIA's code generator fuses `a * b + c` by
  default, which rounds differently in about a quarter of cases. The membrane update
  calls a correctly rounded `fma` on both devices instead.

The CPU kernel skips neurons at rest; the GPU kernel skips resting neurons without
input in the block. Neither changes a result, because such a neuron stays exactly at
rest. Section 2b of VALIDATION.md compares the devices on random networks, on the
whole connectome and frame by frame in the closed loop.

GPU setup: `pip install -e ".[gpu]"` (CUDA 13 driver; use `numba-cuda-mlir[cu12]`
with CUDA 12). numba-cuda-mlir replaces numba-cuda, which NVIDIA has put into
maintenance mode and which does not import with NumPy 2.5. Version 0.5.2 zero-extends
an int32 value in `cuda.atomic.add` into an int64 array; the kernel casts to int64.

### Differences from the reference model

Two deliberate differences in the model, both switchable:

1. **Synaptic weight rescaled.** The reference's only free parameter,
   w_syn = 0.275 mV per synapse, was fitted on FlyWire, which has about 393
   synapses per neuron. MaleCNS has about 745 (different synapse detection, plus
   the nerve cord). `LIFParams.matched_to` scales w_syn so the mean synaptic input
   per neuron is the same as in the fitted model (≈0.145 mV). With 0.275 mV a
   single looming stimulus ignites self-sustaining whole-brain activity.
2. **Neuromodulators have no fast effect.** Dopamine, octopamine and serotonin
   act through slow metabotropic receptors. A handful of these neurons contact
   thousands of partners each; as fast excitatory synapses they broadcast
   activity across the brain. `fast_modulators: true` restores the reference
   convention.

Section 3 of [VALIDATION.md](VALIDATION.md) shows all four combinations.

## Interfaces chosen by hand

A connectome specifies wiring, not how a camera maps onto photoreceptors or how
descending neurons map onto two wheels. These interfaces are the only
hand-designed parts of the brain path and are declared in `brain/malecns.py`.

**Sensory input.** The camera encoder (`vision.py`) computes per-hemifield
features from optical flow. Each feature drives a whole population of visual
projection neurons of the corresponding optic lobe with Poisson spikes:

| feature | neurons driven | why |
|---|---|---|
| `looming_left/right`: expansion inside moving regions | LPLC2 and LC4 of that side | looming-sensitive projection neurons upstream of the escape pathway (Klapoetke et al. 2017; von Reyn et al. 2017) |
| `object_left/right`: motion relative to the background | LC10a of that side | small-object motion detectors used in visual pursuit (Ribeiro et al. 2018) |

Rates are 0 Hz up to a noise floor of 0.1 and rise linearly to 150 Hz. The whole
population is driven together: the model has hemifield resolution, not
retinotopy, and photoreceptors and early motion circuits are bypassed.

**Retinotopic input (experimental, `vision: retina`).** Instead of optical-flow
features, each of the 1,767 lamina L2 cells with column coordinates receives
Poisson spikes in proportion to how much darker its pixel became, and motion would
have to be computed by the fly's own circuits. Where each column looks is derived
from the wiring: T4 cells receive Mi9 input from the side a preferred-direction
stimulus approaches from and Mi4 input from the other side (Takemura et al. 2017),
so their Mi4-minus-Mi9 input centroids give each eye's front-to-back (T4a) and
upward (T4c) axes; opposite subtypes come out 178° and 166° apart and orthogonal
subtypes 87–88°. The camera placement of the eye is a hand-made approximation
(anterior columns at the image midline, posterior at the edge). In this model the
signal reaches Tm1, Tm2 and Tm4 but not T5, T4, HS or LPLC2, even with every L2 cell
at 150 Hz: two GABAergic CT1 cells, point neurons here but local processors in the
fly, inhibit every T5, and Tm9, T5's largest excitatory input, is not driven because
L3 lacks column coordinates in the left eye. The ON pathway cannot be driven,
because glutamatergic L1 signals light by releasing less transmitter. Section 5b of
VALIDATION.md has the numbers; vision therefore stays on optical flow.

**Body input (experimental, off by default).** With `body_feedback: true`, the
tilt of the robot, from its accelerometer, drives Johnston's organ neurons, the
fly's antennal gravity and wind sensors. Tilt features rise linearly from 0 at 3°
to 1 at 20°; overlapping drives add and are capped at 150 Hz.

| tilt | neurons driven | why |
|---|---|---|
| nose up | JO-C (CM, CL, CA) both sides | JO-C respond to the antenna pushed toward the head (Kamikouchi et al. 2009; Yorozu et al. 2009; Patella & Wilson 2018) |
| nose down | JO-E (EV, ED) both sides | JO-E respond to the antenna pulled away |
| left or right side down | JO-C of the lower side, JO-E of the upper side | |

Which body tilt pushes or pulls the antenna is **not established in the
literature**; the table is a hypothesis about the mechanics of a fly's antenna
under gravity. A screen of ten JO types finds 22 JO-type → descending-type pairs
with mirror-image responses that vanish on shuffled wiring (VALIDATION.md §4c),
none of them in the motor read-out. It is off by default because in the closed loop (§4d) a 20° roll
still reaches the Giant Fiber or DNa02 through the connectome and changes motor
commands in some brain seeds (left roll: escape threshold in 3 of 5). On a robot, check `imu_axes` first (HARDWARE.md).

**Motor read-out.** Firing rates are smoothed with a 100 ms time constant.

| read-out | neurons | use |
|---|---|---|
| escape | DNp01 (Giant Fiber), either side | rate / 60 Hz → escape activity; the decoder reverses at ≥ 0.3 |
| steering | DNa01 + DNa02, left minus right | ipsilateral turning (Rayshubskiy et al. 2020); 60 Hz difference = full turn |
| forward | none | constant `cruise` setting in the controller |

DNp02, DNp04, DNp11, DNp09, MDN, DNp18 and DNp20 are logged but do not drive
motors.

The interfaces were chosen after an exploratory screen of 33 visual projection
types (appendix of VALIDATION.md). The screen, the pathway checks and the scene
tests all include shuffled-wiring controls: the same neurons, transmitter signs,
synapse counts and in/out-degrees, with synaptic targets permuted. Behaviour that
survives shuffling is not attributed to the connectome.

## Running

```bash
pip install -e ".[connectome]"
python -m flybrain_robot.connectome.build download --data data/malecns-v1.0
python -m flybrain_robot.connectome.build compile --data data/malecns-v1.0
python -m flybrain_robot.main --backend malecns --data data/malecns-v1.0 --synthetic --dry-run
python -m flybrain_robot.main --backend malecns --data data/malecns-v1.0 --synthetic --wiring shuffled --dry-run
```

Download is about 1.1 GB; compilation takes about 30 s with a measured peak of 4.3 GB
working set (6.1 GB committed). The compiled graph takes 263 MB on disk. Each camera frame advances the brain by one
frame period (33 ms); with strong visual input that takes longer than 33 ms on a
desktop CPU, and the status line prints the ratio as `brain_x_realtime`.

## Limits

- Leaky integrate-and-fire neurons with one weight per synapse count are a
  strong simplification: no graded potentials, gap junctions, neuromodulation,
  plasticity, dendritic computation or measured synaptic strengths. Transmitter
  identity is predicted from EM, not measured.
- The fly's own motion vision (photoreceptors → T4/T5 → lobula plate) is not
  simulated from pixels; a flow estimate is injected into projection neurons.
- No legs or proprioception. Tilt can enter through Johnston's organ, as an
  experimental option with a hypothetical mapping; by default IMU telemetry is
  used by the safety watchdog only.
- Synthetic scenes are not a physical test. See [HARDWARE.md](HARDWARE.md).
