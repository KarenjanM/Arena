# Handover profile: evidence and parameter provenance

Literature audit: 16 November 2026. This records evidence before adding automatic
object-size-to-grip calculations. It does not introduce new runtime parameters.

## What the project PDF asks for

*Handover Receiver Animation*, p. 1, step 6, asks for object size, grip axis and mass
in YAML, with thresholds sourced through Ortenzi et al. Page 2, validation item 4,
asks for comparison with published pose ranges.

The reference is Ortenzi et al., *Object Handovers: A Review for Robotics*, IEEE
Transactions on Robotics 37(6), 1855–1873, 2021,
[published article DOI](https://doi.org/10.1109/TRO.2021.3075365).
The full text inspected here is the later author version,
[arXiv v2, 7 January 2022](https://arxiv.org/html/2007.12952v2).
PDF page references below refer to that version, rather than journal pagination.

## What was found in the review

Section III-B (PDF p. 5) discusses task-dependent grasp selection. Section III-D
(PDF p. 6) discusses shared reachability, approximate midpoint exchanges and
body-dependent target selection. Section VII-A.1 argues for adapting the location
to the person. Section IV-A concerns physical grip forces.
[Ortenzi review](https://arxiv.org/html/2007.12952v2).

**Audit conclusion:** these sections provide no ready-to-use numerical table for
our YAML: no universal handover height/distance bounds, mass/size cutoff for two
hands, palm-angle or torso-bend thresholds, or diameter-to-closure coefficients.
The review is a route to primary experiments; the PDF's wording alone does not
establish that such constants exist.

For our project, an animation closure scalar is a separate quantity from grip
force. A force result cannot be converted directly into `Channel.grip`.

## Concrete numbers that do exist in primary papers

These are examples with their original meanings. None is adopted as a new default.

### Transfer height and distance: Nemlekar et al., 2019

*Object Transfer Point Estimation for Fluent Human-Robot Handovers*,
[author-hosted primary PDF](https://users.wpi.edu/~zli11/papers/C2019_ICRA_Nemlekar_Handover.pdf),
[DOI](https://doi.org/10.1109/ICRA.2019.8794008).
This is review reference [129].

| Reported value | Meaning and source locator | Applicability |
| --- | --- | --- |
| **116 ± 20 cm** | Human–robot separation allowed by the study's positioning boxes; §II, PDF p. 2 / proceedings p. 2628 | Experimental setup, not preferred hand-to-object reach. |
| **1.182 m** | Mean transfer height above the ground in the authors' proposed Gaussian model; §II-A, PDF p. 3 / p. 2629 | An observed/modelled mean, not a lower/upper bound or universal elbow-height rule. |
| **2.91 cm** | Difference between the mean human-to-transfer distance and half the mean human–robot separation; §II-A, PDF p. 2 / p. 2628 | Describes this sample, not a permitted target-position error. |

**Role limitation:** the main study has 20 human **givers** and a robot **receiver**.
Its height mean must not be treated as a measured preference of our human receiver.
The paper itself relates the mean height to its robot receiver.

### Explicit algorithm thresholds: Ardón et al., 2021

*Affordance-Aware Handovers with Human Arm Mobility Constraints*,
[author-hosted primary PDF](https://paolaardon.github.io/files/papers/RA_L_2021.pdf).
This is an additional primary paper, not a table reproduced from Ortenzi.

| Value | Definition in the paper | Classification |
| --- | --- | --- |
| **0.75 m** | Maximum distance from the current human hand to the proposed human grasp; §III, Eq. (4), PDF p. 3 | Reachability-cost cutoff chosen for that method. |
| **0.75 m** | Body-to-transfer distance for comparison method A; §IV-A, PDF p. 3 | Fixed comparison configuration; different distance from Eq. (4). |
| **0.50 m** | Transfer distance for comparison method B; §IV-A, PDF p. 3 | Another comparison configuration, not the lower endpoint of a recommended range. |
| **0.05 m** | Cutoff in the safety-cost terms; §III, Eq. (3), PDF p. 3 | Model-specific penalty threshold, not a certified safety distance or finger-contact tolerance. |

The authors study mobility-dependent preferences using an online survey of 259
participants (§IV). These constants can motivate named comparison cases, but
cannot establish one preferred distance for every person.

### Grasp types: Feix et al., 2016, and Cini et al., 2019

Feix et al., *The GRASP Taxonomy of Human Grasp Types*, classifies **33** static,
stable **one-handed** grasp types, including power, intermediate and precision
categories. See abstract, §II-A and §IV in the
[author-hosted primary PDF](https://www.eng.yale.edu/grablab/pubs/Feix_THMS2016.pdf),
[DOI](https://doi.org/10.1109/THMS.2015.2470657).
This is review reference [77]. It supplies a vocabulary, not a calibrated table
mapping object diameter or mass to our closure scalar or to bimanual selection.

There are numerical descriptions: §IV-D.1, printed p. 71 / PDF p. 6, reports that
most grasp types in its source work datasets involve objects **below 500 g** and
hand apertures **5 cm or less**. These describe observed use in those environments;
they are not cutoffs for switching grasp type or using two hands. Neither number
calibrates our finger-curl scalar. The prototype bottle's 0.5 kg value was not
derived from that observation.

Cini et al., *On the choice of grasp type and location when handing over an
object*, reports that **passers** favor precision grasps and leave handles
available for the receiver in its task-constrained experiment. In-hand and
bimanual manipulation were excluded. See the
[primary article's abstract](https://doi.org/10.1126/scirobotics.aau9757)
([accessible abstract](https://pubmed.ncbi.nlm.nih.gov/33137738/)).
This result must not become a rule that our **receiver** always uses a precision
grasp, or a numerical cutoff for using two hands.

## Measurement definitions to preserve

Before importing any number, record the actor role, object, posture, reference
frame, measured landmarks, units, participant sample, and whether the number is
a condition, observation, fitted coefficient, or algorithm threshold.

In particular, keep these quantities separate:

- Human–robot separation.
- Distance from the human body to the transfer point.
- Distance from the current hand to a candidate grasp.
- Height above the floor versus a height normalized by an anatomical landmark.
- Physical thumb–finger opening, finger joint angles, grip force, and our
  dimensionless closure value.

Our profile's `distance` is the **forward coordinate of a suggested object-centre
point** in the pedestrian floor frame. A lateral offset makes this differ from the
full horizontal body-to-point distance. A published distance requires a frame and
landmark conversion before comparison with that coordinate.

## What the current implementation's values actually mean

The following entries describe our code, not findings attributed to the papers.

| Current parameter or rule | Provenance/status |
| --- | --- |
| Bottle size `(0.07, 0.07, 0.22)` m, mass `0.5` kg; box size `(0.25, 0.30, 0.20)` m, mass `1.0` kg | Explicit prototype examples in [objects.yaml](objects.yaml); no measurements or literature thresholds attached. |
| Bottle uses one hand; box uses two | Explicit object metadata. No mass/width cutoff is implemented. Mass currently does not affect the pose. |
| `grip_axis == "z"` chooses `thumb_up`; other axes choose `palm_up` | Prototype orientation rule in [handover_profile.py](handover_profile.py). These names describe wrist orientation, not taxonomy grasp types. |
| Suggested height = estimated elbow height plus a palm or half-object-height offset | Upright-body geometric heuristic; no empirical preferred-height fit. |
| Forward distance = forearm length + palm offset + half object x-size | Geometric heuristic; no empirical preferred-distance fit. |
| Palm offset = `0.45 * hand length` | Shared approximation from [support.py](../../../support.py), reused for consistency; no handover threshold is claimed. |
| `grip_weight` in `0..1` | Prototype take presets in [objects.yaml](objects.yaml): `0.6` for the bottle and `0.2` for the box. Profiles without a value use `0.6`. |
| Fixed `GRIP` curls, fade `0.25` s | Finger animation interpolated in [the gesture layer](../../__init__.py) and [hands.py](../../../hands.py). The scalar sets how far the fingers curl toward the preset. |

The handover path in Receive uses approximate palm-centre targets offset from the
commanded object centre. The geometric solve uses measured arm lengths and joint limits.

## Consequence for the next implementation step

For a first cylinder-diameter-to-closure mapping, select two rig-specific reference
poses and document their diameters, body/hand size, closure values and calibration
procedure. Until measured against hand/object geometry or empirical data, label
the mapping a **prototype animation heuristic**. The literature above does not
supply its two coefficients.

Keep calculating the suggested closure separate from requesting the take phase.
Retain explicit one-/two-hand metadata until a justified selection model is chosen.

For evaluation, compare like-for-like quantities in named study conditions. Treat
0.50 m and 0.75 m as possible comparison configurations with a declared distance
definition, rather than declaring a general 0.50–0.75 m acceptable range. A single
reported mean such as 1.182 m supplies neither a distribution spread nor bounds
for a randomized human population.

Suggested clarification of the project requirement: **use Ortenzi to identify
primary studies, source each adopted parameter with its definition and context,
and label unsourced choices as project heuristics.**
