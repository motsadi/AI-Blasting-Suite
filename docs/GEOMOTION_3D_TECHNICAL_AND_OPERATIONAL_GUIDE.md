# GeoMotion 3D Engine

## Technical and Operational Study Guide

**Document status:** Synthetic Demonstration / Uncalibrated — Planning Only
**Engine version described:** GeoMotion 0.2 Event Physics
**Primary application:** Open-pit diamond-mine blast movement and ore-control studies

---

## 1. Purpose of GeoMotion

Blasting fragments a bench but also moves ore and waste away from their original positions. If dig limits are based only on the pre-blast block model, loaders can:

- leave valuable ore behind as apparent waste;
- send waste to the plant, creating dilution;
- mix geological facies and grades;
- deliver a feed grade different from the planned grade; and
- lose the relationship between grade-control information and the material actually loaded.

GeoMotion estimates a three-dimensional movement vector for material cells in the blast. It then moves their tonnes, grade, facies and contained carats into a post-blast model. The intended operational outcome is better post-blast mark-out and improved decisions about what is sent to the plant, stockpile or waste dump.

GeoMotion does **not** fire a blast, program detonators or replace a qualified blasting engineer, geologist, surveyor or site procedure.

## 2. Current maturity

GeoMotion currently contains a reduced-order event-physics solver and a synthetic diamond geology model. It is useful for:

- demonstrating the complete data workflow;
- studying timing and parameter sensitivity;
- testing visualization and reporting;
- defining the mine data that must be collected; and
- preparing a calibration programme.

It is not yet a validated production movement predictor. Until measured mine datasets are loaded and the model is calibrated, recovery, dilution, grade and uncertainty are synthetic planning outputs.

This distinction is essential:

| Capability | Current status |
|---|---|
| Delay-bearing tie-up import | Implemented |
| 1 m³-cell physics calculation (1 m × 1 m × 1 m) | Implemented |
| Timed hole/deck event sequence | Implemented |
| S135B reduced-order energy model | Implemented with assumptions |
| Dynamic relief approximation | Implemented |
| Conservative tonnes/carat transport | Implemented |
| Solid 3D voxel visualization | Implemented |
| Real grade-control block model | Adapter defined; synthetic provider remains active |
| Measured pre/post surfaces | Adapter defined; not yet applied to the solver |
| Movement-monitor calibration | Adapter defined; no site model trained yet |
| Site-validated AI correction | Not yet available |
| FEM/DEM detonation hydrocode | Not part of this product |

## 3. End-to-end workflow

1. Import a charged-hole CSV containing geometry, charge and cumulative firing time.
2. Validate coordinates, depths, charges and unique timing.
3. Normalize firing time so the first event occurs at zero milliseconds.
4. Add hole/deck, explosive, rock, joint and ore-control assumptions.
5. Construct contiguous 1 m³ source cells (1 m × 1 m × 1 m).
6. Process holes/decks in actual sampled firing order.
7. Apply energy, pressure/impulse, burden velocity, confinement and relief effects.
8. Advance voxel movement through the firing sequence.
9. Settle and conservatively remap material to unique post-blast cells.
10. Recalculate ore/waste classification, recovery, dilution, grade and mixing.
11. Review uncertainty, event diagnostics and 3D visualizations.
12. Export interactive-level or full-resolution movement records.

## 4. Required tie-up data

The required columns are:

| Field | Unit | Meaning | Why it matters |
|---|---:|---|---|
| Hole ID | text | Unique hole identifier | Event traceability, QA and reporting |
| X | m | Collar easting | Blast geometry and distance calculations |
| Y | m | Collar northing | Blast geometry and distance calculations |
| Z | m RL | Collar elevation | Bench and deck vertical geometry |
| Depth | m | Total drilled depth | Toe position and source volume |
| Charge | kg | Total explosive mass | Event energy and impulse |
| Delay | ms | Cumulative nominal firing time | Firing order, relief and interaction |

Hole ID can be generated when absent, but site IDs are strongly preferred.

### 4.1 Timing interpretation

GeoMotion expects cumulative firing times, not inter-hole increments. If a dataset ranges from 8,000 to 11,120 ms, GeoMotion preserves those values for audit but simulates:

`normalized time = original delay - minimum original delay`

The resulting simulation range is 0 to 3,120 ms.

Every hole must have a unique valid delay. GeoMotion does not invent timing for this module because timing controls when relief becomes available to later holes.

### 4.2 Input QA

The importer checks:

- blank or invalid coordinates;
- missing or non-positive charge;
- missing or implausibly shallow depth;
- missing or duplicate firing times;
- duplicate Hole IDs;
- near-overlapping collar coordinates;
- trailing empty rows;
- inferred floor RL from `Z - Depth`; and
- median nearest-hole spacing.

Invalid depth/charge rows are explicitly excluded and reported. They are never silently converted into valid blast holes.

## 5. Optional hole and charge data

For higher-quality modelling, each hole should also contain:

| Input | Preferred data |
|---|---|
| Diameter | Actual drilled diameter by hole |
| Inclination | Angle from vertical |
| Hole azimuth | Direction of inclined hole |
| Stemming | Actual stemming length and material |
| Decks | Top depth, bottom depth, explosive and mass for each deck |
| Primer | Type, mass and position in each deck |
| Charge density | Actual in-hole product density |
| Actual firing time | Measured firing time when available |

When deck data are absent, GeoMotion assumes one continuous toe-up S135B deck, stated average stemming and one 400 g Pentolite booster near the toe. That assumption is marked synthetic.

## 6. S135B explosive model

The current profile uses:

- product: S135B bulk emulsion;
- density: 1,250.51 kg/m³;
- supplied relative weight strength: 115%;
- nominal VOD: 4,500 m/s;
- uncertainty range: approximately 3,500–5,500 m/s;
- primer: 400 g Pentolite booster; and
- electronic initiation.

The geometric linear-charge check is:

`linear charge = π × (diameter / 2)² × explosive density`

This gives approximately:

| Hole diameter | Calculated loading |
|---:|---:|
| 127 mm | 16 kg/m |
| 165 mm | 27 kg/m |
| 250 mm | 61.4 kg/m |

### Important limitation

GeoMotion does not have product-certified S135B Jones-Wilkins-Lee constants or cylinder-test data. Its detonation pressure is a proxy:

`pressure proxy ≈ 0.25 × explosive density × VOD²`

Chemical energy is estimated from explosive mass, an ANFO reference energy and RWS. Only a small bounded fraction is allocated to coherent bulk rock movement. Most energy is understood to be consumed by fragmentation, heat, gas expansion, vibration and other mechanisms.

The model must therefore be calibrated against measured burden velocity, movement vectors and final muckpile shape.

## 7. Electronic timing scatter

Electronic detonators are precise but not perfectly exact. GeoMotion samples actual firing time around nominal time using:

`standard deviation = 0.094 + 0.000345 × normalized nominal delay`

The random realization is repeatable from the project seed. Event outputs report nominal time, actual sampled time and timing error.

Manufacturer-specific timing accuracy should replace this generic relationship when available.

## 8. Rock and structure inputs

The editable defaults include:

| Parameter | Current default | Physical role |
|---|---:|---|
| Rock density | 2.35 t/m³ | Voxel tonnes and inertia |
| UCS | 120 MPa | Resistance and energy transfer |
| Tensile strength | 10 MPa | Movement-energy partition |
| Young’s modulus | 55 GPa | Rock impedance |
| Poisson ratio | 0.24 | Stored for expanded constitutive modelling |
| Damping ratio | 0.28 | Velocity decay and settlement |
| Fragmentation index | 0.55 | Growth of released volume |
| Joint dip | 70° | Structural orientation |
| Joint direction | 90° | Directional anisotropy |
| Joint spacing | 2.5 m | Structural scale |
| Joint persistence | 0.60 | Strength of anisotropic effect |
| Swell factor | 1.25 | Broken-rock volume response |

These are not measured mine properties. Laboratory tests, geotechnical domains and structural mapping should replace them.

## 9. One-cubic-metre cell model

The source volume is divided into contiguous cells measuring 1 m × 1 m × 1 m. Each cell therefore has a volume of exactly 1 m³ and contains:

- source X, Y and Z;
- density and tonnes;
- geological facies;
- diamond grade in cpht;
- ore/waste class;
- contained carats; and
- provenance.

Cell tonnes are:

`tonnes = density (t/m³) × cell volume (1 m³)`

Contained carats are:

`contained carats = tonnes × grade cpht / 100`

The current facies and grade distribution are seeded synthetic geology. They demonstrate material tracking but must be replaced with the mine’s grade-control block model.

## 10. Event-physics engine

### 10.1 Event scheduling

Each hole or deck becomes an event. Events are sorted by sampled actual firing time. The engine records the deck centre, charge, primer, diameter, stemming and event geometry.

### 10.2 Spatial influence

A three-dimensional spatial index identifies cells close enough to be affected by each event. Influence decreases with distance. This avoids calculating every hole against every cell and makes approximately 200,000-cell simulations practical.

### 10.3 Pressure, impulse and burden velocity

For each event, GeoMotion estimates:

- detonation-pressure proxy;
- total chemical energy;
- movement-energy fraction;
- stemming effectiveness;
- gas-pressure decay time;
- local impulse velocity; and
- representative burden velocity.

A cell’s approximate speed increment follows an energy relationship:

`speed ≈ sqrt(2 × allocated movement energy / local voxel mass)`

The result is modified by attenuation, confinement, relief, rock strength, impedance and joint orientation.

### 10.4 Dynamic free-face opening

The initial free-face azimuth establishes the preferred relief direction. Every event updates a release field around affected voxels. Later events therefore experience different confinement from early events.

This reduced-order release field represents the operational concept that a hole firing into confined rock behaves differently from a hole firing after earlier rows have opened relief.

### 10.5 Movement through time

Between firing events:

- cell positions advance according to current velocity;
- velocity decays according to damping;
- later events add new impulse; and
- effective timing is accumulated for each voxel.

After the final event, the model applies bounded settlement, gravity and swell. Movement is capped to prevent unphysical runaway values in an uncalibrated simulation.

### 10.6 Conservative remapping

Independent trajectories can propose overlapping destination cells. GeoMotion resolves these collisions by settling material into unique destination columns.

The remap preserves:

- total tonnes;
- grade carried by each source cell;
- facies;
- ore/waste source identity; and
- contained carats.

A zero mass-balance error means the numerical transport did not create or destroy material. It does **not** prove the predicted movement is accurate.

## 11. How the AI works

### 11.1 Current implementation

The current authoritative model is primarily event physics plus repeatable uncertainty sampling. It does not yet contain a site-trained AI model that has learned Orapa movement.

The term “hybrid” currently means that the physics realization includes sampled timing and VOD uncertainty. It must not be represented as a validated AI prediction.

### 11.2 Intended site-calibrated AI

Once measured blasts are available, the recommended AI is a residual model:

`final movement = physics prediction + learned site residual`

The physics model supplies a constrained baseline. The AI learns systematic errors caused by geology, structures, explosive behaviour and local operational conditions.

Candidate AI inputs include:

- physics-predicted dx, dy and dz;
- depth within bench;
- distance and direction to free face;
- burden and spacing;
- charge, charge length and energy;
- nominal and actual timing;
- local timing interactions;
- stemming effectiveness;
- rock domain, density, UCS and modulus;
- joint orientation and distance to structures;
- source grade/facies;
- pre/post surface differences; and
- neighboring measured movement.

Training targets should be measured `dx`, `dy`, `dz`, final surface elevation and, where available, destination classification.

### 11.3 Correct validation

Cells from one blast are highly correlated. Randomly splitting cells would leak information and exaggerate model accuracy. Training and validation must be split by whole blast:

- train on historical blasts;
- validate on unseen blasts;
- retain a final blind test set; and
- report performance by geological and blast domain.

Required metrics include vector MAE, RMSE, directional bias, vertical/horizontal error, surface error, recovery/dilution reconciliation and uncertainty coverage.

## 12. Outputs

### 12.1 Per-voxel or visualization block

- source and destination coordinates;
- dx, dy and dz;
- velocity components;
- displacement magnitude;
- uncertainty;
- peak impulse;
- burden velocity;
- contributing event;
- facies and grade;
- source and destination class;
- tonnes and contained carats; and
- provenance.

### 12.2 Per-event output

- hole and event index;
- nominal and actual firing time;
- timing error;
- charge and sampled VOD;
- pressure proxy;
- chemical and movement energy;
- gas time constant;
- stemming effectiveness;
- burden velocity; and
- released voxel count.

### 12.3 Ore-control KPIs

**Ore recovery**

`source ore tonnes ending in ore destination / source ore tonnes × 100`

**Ore loss**

`source ore tonnes ending in waste destination / source ore tonnes × 100`

Recovery plus ore loss should be approximately 100%.

**Dilution**

`source waste tonnes entering ore destination / total post-blast ore-stream tonnes × 100`

**Carat recovery**

`contained source-ore carats retained in ore destination / initial source-ore carats × 100`

**Predicted feed grade**

`carats in post-blast ore stream × 100 / post-blast ore-stream tonnes`

### 12.4 Mixing matrix

The matrix reports:

- ore to ore;
- ore to waste;
- waste to ore; and
- waste to waste.

Ore-to-waste is loss. Waste-to-ore contributes dilution.

### 12.5 Loader-scale metrics

One-cubic-metre geological-cell selectivity is not achievable by a production loader. GeoMotion groups destination cells into a configurable minimum mining unit and calculates loader-scale recovery and dilution.

This distinction prevents presenting fine voxel selectivity as an operationally achievable result.

## 13. Visualization

The interface provides:

- in-situ, movement and post-blast views;
- joined cubes or 8–22% seams with visible cube edges;
- default gold ore / stone waste colouring, plus facies, grade, displacement, uncertainty, burden-velocity and impulse colours;
- live legend counts;
- optional source-to-destination vectors;
- plan, section and perspective cameras that fit the whole model;
- clipping for internal sections;
- 1× vertical scale by default; and
- event timeline playback.

The 3D workspace draws the block model only. It does not render a bench, highwall, free-face arrow or mine context.

The backend computes 1 m³ cells (1 m × 1 m × 1 m). Interactive responses may aggregate contiguous cells into larger level-of-detail cubes for performance. Full-resolution records remain available from the authoritative compressed export.

If the interface warns that it is using a coarse 3 m browser preview, the upgraded Cloud Run endpoint is unavailable. That preview must not be described as the full engine.

## 14. Data required for a reliable mine model

### 14.1 Minimum per blast

- final as-drilled collars, toes, depth, inclination and azimuth;
- final as-charged decks, explosive mass, density and stemming;
- final programmed delay and initiation sequence;
- bench limits, free faces and pre-blast topography;
- grade-control block model with coordinate system;
- geological domains, density, facies and grade;
- post-blast drone or LiDAR surface; and
- final dig limits.

### 14.2 Strongly recommended calibration data

- Blast Movement Monitor or surveyed marker vectors;
- measured burden velocity/video where practical;
- measured VOD;
- electronic detonator timing records;
- fragmentation measurements;
- mapped joints and major structures;
- loader polygons or bucket/truck destinations;
- stockpile, plant and waste routing;
- truck payloads and dispatch records; and
- plant/feed reconciliation.

### 14.3 Explosive/manufacturer data

- current S135B technical data sheet;
- density by product batch or truck;
- VOD versus diameter and density;
- relative weight and bulk strength definitions;
- gas volume and energy;
- approved pressure/EOS information where available;
- booster specification; and
- electronic detonator accuracy specification.

### 14.4 Rock-mechanics data

- density by domain;
- UCS and tensile strength;
- Young’s modulus and Poisson ratio;
- discontinuity orientation, spacing and persistence;
- weathering and alteration;
- rock quality or blastability index; and
- water conditions.

## 15. Recommended calibration programme

### Stage 1: Data audit

Select historical blasts with complete tie-up, block model, pre/post survey and reconciliation. Standardize coordinate reference systems, units, naming and timestamps.

### Stage 2: Physics calibration

Calibrate broad movement scale and direction using measured vectors and surfaces. Adjust energy partition, attenuation, damping, relief response and structural anisotropy within physically defensible ranges.

### Stage 3: AI residual training

Train on multiple blasts and geological conditions. Keep entire blasts together during splitting. Compare physics-only and hybrid performance.

### Stage 4: Blind validation

Predict unseen blasts before reviewing their post-blast measurements. Establish acceptance thresholds for vector, surface and ore-control errors.

### Stage 5: Controlled operational trial

Run GeoMotion in parallel with existing grade-control procedures. Do not change dig limits solely from GeoMotion until site governance approves the validated workflow.

### Stage 6: Continuous reconciliation

After every blast, load measurements and compare predicted versus actual outcomes. Monitor drift caused by product, geology, design or operating-practice changes.

## 16. How blast teams can use GeoMotion

### Before blasting

- validate charged-hole and timing files;
- identify missing, duplicate or invalid holes;
- inspect firing progression and expected relief;
- compare timing or charge scenarios;
- review sensitivity to free-face direction, stemming, rock strength and VOD; and
- communicate expected movement direction to geology and survey.

GeoMotion must remain decision support. Approved blast-design and firing systems remain authoritative.

### Immediately after blasting

- load the final tie-up and actual timing where available;
- import post-blast survey data;
- calculate a post-blast movement model;
- compare predicted muckpile shape with survey;
- produce candidate post-blast ore-control boundaries; and
- identify high-uncertainty zones requiring conservative mark-out or measurement.

### During loading

- provide geology and load-and-haul teams with post-blast classification;
- apply realistic loader/MMU selectivity;
- flag boundaries with high predicted mixing;
- reconcile truck destinations and tonnes; and
- retain an auditable link between source blocks and destinations.

### After completion

- compare predicted and measured movement;
- reconcile ore, waste, grade and carats;
- investigate loss and dilution mechanisms;
- update site calibration; and
- use trends to improve timing, burden, spacing, stemming and charge practices.

## 17. How GeoMotion can improve recovery and dilution

GeoMotion does not create value merely by producing vectors. Value comes from integrating post-blast information into ore control.

It can improve recovery by:

- relocating ore boundaries after movement;
- identifying ore displaced into apparent waste;
- reducing ore sent to waste dumps;
- highlighting vertical movement that a two-dimensional translation misses; and
- adjusting dig limits to the post-blast material location.

It can reduce dilution by:

- locating waste moved into the ore envelope;
- quantifying mixing along contacts;
- accounting for loader selectivity;
- identifying timing/design conditions associated with excessive cross-contact movement; and
- guiding targeted measurement in uncertain zones.

The benefit must be measured through reconciliation: predicted versus actual tonnes, grade, carats and destinations.

## 18. Governance and acceptance criteria

Before operational use, the mine should define:

- data ownership and sign-off responsibilities;
- approved coordinate systems and units;
- acceptable missing-data thresholds;
- model-version control;
- calibration validity by geological domain;
- maximum acceptable movement and surface errors;
- uncertainty-based decision rules;
- independent review requirements;
- change-control procedures; and
- fallback procedures when required data are unavailable.

Every result should record engine version, assumptions, provenance, input files, seed, calibration version and validation status.

## 19. Key limitations

- Current geology and grade are synthetic.
- Current movement has not been calibrated to measured mine movement.
- Pressure is a surrogate, not a certified S135B JWL solution.
- Fracture formation and fragment contacts are not explicitly solved.
- Registered optional-file metadata does not yet mean the contents affected a run.
- Uncertainty is a sensitivity proxy, not calibrated probability.
- Level-of-detail visualization may use larger cubes than the 1 m³ physics cells.
- Zero mass-balance error confirms conservation, not prediction accuracy.
- Recovery and dilution are only meaningful when real block models and dig limits are used.

## 20. Practical interpretation rule

Always ask three questions when reviewing a result:

1. **What was measured?**
2. **What was supplied by the site or manufacturer?**
3. **What was synthetic or assumed?**

GeoMotion becomes operationally valuable only as the first category grows and the third category is reduced through disciplined data collection and validation.

---

## Glossary

- **Burden:** Distance from a hole or row to the available free face.
- **Spacing:** Distance between adjacent holes in a row.
- **cpht:** Carats per hundred tonnes.
- **Deck:** A discrete explosive column within a hole.
- **Dilution:** Waste entering material classified and mined as ore.
- **Facies:** Geological material subdivision with distinct properties.
- **Free face:** Unconfined surface toward which broken rock can move.
- **Impulse:** Pressure acting over time, producing momentum.
- **MMU:** Minimum mining unit representing operational selectivity.
- **Ore loss:** Ore classified or mined as waste.
- **Residual model:** AI model learning errors remaining after physics prediction.
- **RWS:** Relative weight strength compared with a reference explosive.
- **VOD:** Velocity of detonation.
- **Voxel:** A three-dimensional material cell; GeoMotion physics voxels are exactly 1 m³ (1 m × 1 m × 1 m).
