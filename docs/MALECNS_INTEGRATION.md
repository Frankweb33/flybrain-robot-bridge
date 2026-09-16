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

Two options extend the reference model and are off unless a mode asks for them. Each
neuron has its own resting potential, so a cell can be given a tonic firing rate that
synaptic input grades in both directions (used by `vision: eye`; with one constant for
every neuron nothing changes). And each neuron carries a flag that sends its outgoing
synapses into a second conductance with its own time constant, for inputs that
recordings report as slow; with both constants equal the two pools behave identically.

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
   convention. What this removes is worth stating plainly: 868 neurons, 631,260
   edges and 1,788,712 synapses (1.4% of the total), with a mean out-degree of
   727 against 153 for the brain as a whole. The receptor argument licenses
   "slow", not "zero", and octopamine in particular carries the fly's locomotor
   state into visual processing — which a walking robot removes here.

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
| `object_left/right`: motion relative to the background | LC10a of that side | small-object visual projection neurons that steer pursuit **during courtship** (Ribeiro et al. 2018) |

The LC10a citation is weaker than the others and is used outside its context: Hindmarsh
Sten et al. 2021 show that this pathway's gain is raised by P1-driven arousal and that it
specifies tracking *in the courting male*. This model has no arousal state and no
neuromodulation, so it runs the pathway in the condition where the fly does not use it for
pursuit. LC10a → DNa02 is also entirely polysynaptic here (0 direct synapses; the route
runs through AOTU neurons).

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

**Vision from the photoreceptors (experimental, `vision: eye`).** The camera drives the
5,826 photoreceptors of the two eyes themselves, each at a rate proportional to the
brightness of its pixel, and everything after that is the connectome. Three changes are
needed to make that possible at all, and each is declared:

The retina this drives is not a fly's retina. The release contains 3,377 R1-R6 cells
against about 9,600 in a fly, and they are split 1,112 left against 2,265 right, so one
eye is sampled twice as densely as the other. Any left-right comparison in this mode,
including the pre-registered mirror-symmetry condition, inherits that asymmetry.

1. **Columns from the wiring.** The release assigns optic-lobe columns to 23,720 neurons
   but not to T4, T5, Tm3, L4 or the photoreceptors. A columnar cell's partners sit in and
   around its own column, so the synapse-weighted centroid of the partner columns recovers
   it: measured on the neurons that do have a column, and using only their partners, it is
   exact for 97.7 % and within one column for 99.3 % (`connectome/columns.py`). Wide-field
   cells stay without a column because their partner columns are spread out (2.1 columns
   for LPLC2, 3.0 for LC4, 14.0 for CT1, against 0.8–1.3 for the columnar cells).
2. **CT1 compartments.** One CT1 cell per eye contacts every column, and its compartments
   are electrically isolated in the fly (Meier & Borst 2019). As a single point neuron it
   fired at over 100 Hz and inhibited the whole eye at once, so it is split into one
   neuron per column, keeping every synapse.
3. **Tonic release.** Most optic-lobe neurons do not spike; they release transmitter
   continuously, and photoreceptors are histaminergic, so the fly's lamina signals
   *darkness as a release from inhibition*. A spiking model whose neurons are silent at
   rest cannot express that at all. Columnar optic-lobe neurons therefore get a resting
   potential above threshold and fire at 20 Hz without input; synaptic input grades that
   rate in both directions (`brain/lif.py`, `resting_for_rate`). This surrogate is
   one-sided: 20 Hz corresponds to 0.71 mV above threshold, so 0.71 mV of inhibition
   silences the cell completely while excitation raises the rate without bound. And it is
   not cheap — 72,658 tonic cells at 20 Hz are 1.45 million spikes per second of baseline,
   more than the runaway state that motivated rescaling the synaptic weight (below).

With those, the motion detectors are alive, which they were not in `vision: retina`, and
the sign structure downstream of the lamina is partly the fly's. On a full-field
brightness step every lamina cell depolarises in the dark, as it must when the
photoreceptors are inhibitory (L1 7.5 → 17.1 Hz, L2 2.3 → 5.2, L3 5.2 → 16.7) — that is
one sign inversion, not an ON/OFF split. The split is made downstream, and the model gets
it half right: the OFF motion detectors rise in the dark (T5a 7.1 → 9.2, T5c 13.3 → 15.4)
and the ON ones fall (T4a 9.6 → 6.5, T4c 14.0 → 11.0), as does Tm3 (13.4 → 8.7). But Mi1,
the other canonical ON cell, comes out the wrong way (4.0 → 4.5), so the polarity of the
ON pathway is not reproduced at its first stage. L1 feeds the ON pathway and L2/L3 the OFF
one; the lamina rates above say nothing about which is which.

Direction selectivity is not. Two pre-registered checks failed (§5c): the direction
selectivity index of every T4 and T5 subtype is 0.00, indistinguishable from the shuffled
controls, both with one synaptic time constant for every synapse and with the 50 ms
constant that recordings report for the slow inputs Mi9, Tm9, Mi4 and CT1 — the latter
suppresses T4 and T5 about tenfold (T4a of the left eye 8.9 → 0.7 Hz) without making them
selective. With identical kinetics that is what theory predicts: a drifting grating then
delivers the same input in both directions however the cells are wired. The connectome
supplies the spatial offsets (§5b) but not the time constants and nonlinearities that turn
them into direction selectivity, so vision stays on optical flow.

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

Three things about this read-out are engineering, not biology, and a reader should have
them in front of the numbers. The Giant Fiber commands an escape **takeoff** in the fly,
not backward walking; backward walking is commanded by MDN, which is in this model and is
not used, so mapping the Giant Fiber onto a reverse is a choice made for a wheeled robot.
The model's Giant Fiber also fires 127 spikes during one approach where the fly fires
about one action potential (VALIDATION.md §5f), so the read-out reads a rate from a cell
that does not rate-code. And the two scale constants, 60 Hz for a full turn and 60 Hz for
full escape, are fitted to the robot, not taken from a recording: Rayshubskiy et al.
support the sign and the linear relation between the left-right DNa02 difference and
turning, not an absolute rate.

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
- The fly's own motion vision is simulated from pixels only as an experiment
  (`vision: eye`): the photoreceptors reach T4 and T5 with the right ON/OFF
  polarity, but the motion detectors show no direction selectivity, so by default
  a flow estimate is injected into projection neurons instead. Every synapse of
  this model shares one time constant and one weight per synapse count, and
  direction selectivity needs input-specific kinetics, which a connectome does
  not contain.
- No legs or proprioception. Tilt can enter through Johnston's organ, as an
  experimental option with a hypothetical mapping; by default IMU telemetry is
  used by the safety watchdog only.
- The senses are empty and the body is not there. 17,901 sensory neurons are
  simulated at zero: no olfaction, no taste, no mechanosensation, no wind, no leg
  load, and in `flow` mode no photoreceptors either, since the camera enters at
  the visual projection neurons. The 20,239 nerve-cord neurons include 708 motor
  neurons that are simulated and then ignored, because there are no muscles and
  no body to drive; the motor command is read from three descending types
  instead. Every threshold in this report is therefore calibrated in a brain that
  receives one sense and moves nothing.
- Synthetic scenes are not a physical test. See [HARDWARE.md](HARDWARE.md).
