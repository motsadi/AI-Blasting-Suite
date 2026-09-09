# GeoMotion 3D Engine

GeoMotion 3D is an independent, reduced-order event-physics blast-movement module. It transforms a delay-bearing charged-hole tie-up and a contiguous 1 m³ mining block model into a mass-conserving post-blast material model. Every physics cell is 1 m × 1 m × 1 m.

## Status and safety

All current geology, movement calibration, grade, recovery, dilution and uncertainty outputs are synthetic unless their provenance explicitly says measured. They are not validated predictions for Orapa or any other mine and must not be used for field execution, dig-limit control, resource reporting or production decisions.

The module does not arm, program or communicate with detonators. `SAFETY_SCOPE.md` remains in force.

## Required tie-up

| Field | Requirement | Notes |
|---|---|---|
| Hole ID | Recommended | Missing IDs receive generated identifiers |
| X, Y | Required | Projected coordinates in metres |
| Z | Required | Collar RL |
| Depth | Required | At least 1 m |
| Charge | Required | Positive kilograms |
| Delay | Required | Unique cumulative nominal firing time in milliseconds |

The importer preserves cumulative delay and simulates `Delay - minimum Delay`. It rejects missing or duplicate delays rather than inventing a timing pattern. Invalid charge/depth rows are explicitly excluded and reported.

## S135B, decks and rock defaults

- S135B density: `1250.51 kg/m³` and RWS `115%` (site supplied).
- VOD: nominal `4500 m/s`, sampled within an assumed `3500–5500 m/s` range.
- Pentolite booster: `400 g`.
- Linear loading checks: approximately 16, 27 and 61.4 kg/m for 127, 165 and 250 mm holes.
- Missing deck records: one continuous toe-up S135B deck and one booster near the toe, marked synthetic.
- Electronic timing scatter: `σ = 0.094 + 0.000345 × normalized delay` ms unless overridden.

Editable synthetic rock defaults include UCS, tensile strength, Young's modulus, Poisson ratio, damping, fragmentation index and joint orientation/spacing/persistence. These are sensitivity assumptions, not measured mine properties.

## Optional mining block model

GeoMotion can run from the delay-bearing tie-up alone by generating a clearly labelled simulated 1 m³-cell block model around the blast. This supports workflow evaluation before mine geology is available; its recovery and dilution outputs are uncalibrated.

When available, users can upload a CSV block model containing `X,Y,Z,Density`, with `Block ID,Grade,Facies` recommended. Optional `Size X,Size Y,Size Z` fields must each equal 1 m, giving a block volume of exactly 1 m³; other dimensions are rejected with an instruction to resample. Tonnes are calculated as `density (t/m³) × block volume (1 m³)`. The model's measured density, grade and facies then replace the simulated geology and drive tonnes, ore/waste classification, contained grade, movement and mixing outputs.

Here, 1 m³ means one cubic metre, not one cubic centimetre. A 1 cm³ mine model would require 1,000,000 cells per cubic metre and is not computationally practical at blast scale.

## Other optional measured datasets

The UI can also register geological structures, pre/post-blast surfaces, movement monitors, dig limits and loader/MMU geometry. Backend CSV adapters validate:

- Block model: `X,Y,Z,Density,Grade,Facies`, with optional block dimensions.
- Surface: `X,Y,Z`.
- Movement monitors: `X,Y,Z,dX,dY,dZ`.
- Dig limits: polygon ID, vertex sequence, `X,Y`, destination.

Registered metadata alone does not alter a simulation. The uploaded mining block model is the exception: its validated contents directly replace the synthetic geology provider.

## Contiguous 1 m³ source model

The engine creates mass-bearing cells with dimensions `1 m × 1 m × 1 m` and volume `1 m³` around the drilled footprint without a vertical-level cap. Each cell carries source/destination coordinates, facies, density, tonnes, synthetic cpht grade, classification and contained carats.

## Event-physics model

For every hole/deck event in sampled firing-time order, the solver:

1. Calculates an S135B chemical-energy and detonation-pressure proxy.
2. Applies a bounded bulk-movement energy partition and stemming effectiveness.
3. Queries nearby voxels using a spatial index.
4. Calculates attenuation, confinement, rock impedance, joint anisotropy and current relief.
5. Applies a pressure/impulse-derived velocity increment and records burden velocity.
6. Advances position between events with damping.
7. Updates a release field so later holes respond to newly opened relief.
8. Applies bounded settlement, gravity and swell.

This is not a detonation hydrocode. Product-certified JWL constants, explicit fracture mechanics and fragment contacts require manufacturer cylinder tests and FEM/DEM software.

The result is conservatively remapped into unique destination columns. Collisions settle vertically; tonnes, facies, grade and contained carats are preserved. Loader/MMU classification is calculated separately from 1 m³-cell ore-control classification.

## Visualization and transport

The backend computes every 1 m³ cell. Interactive responses may aggregate contiguous source cells into larger regular level-of-detail cubes while preserving aggregate tonnes and carats. The UI uses GPU-instanced solid cells with visible cube edges.

The 3D workspace renders the block model only. It does not draw a bench floor, highwall, free-face arrow, north arrow, hole markers or a synthetic whole mine. Default colouring is ore (gold) versus waste (stone). Fit frames the full source-to-destination AABB so every cube is in view.

Controls include 8–22% cube seams, model cutaway, 1× vertical scale by default, plan/section/perspective cameras, event playback, optional movement vectors, grade/facies, ore/waste, displacement, uncertainty, burden velocity and peak impulse. Full-resolution gzip CSV export is available from the authoritative backend.

If Cloud Run returns `404` or `405`, the UI runs a clearly labelled coarse browser preview. That preview is not the authoritative 1 m³-cell event solver.

## Outputs

- Per voxel/LOD block: origin, destination, `dx/dy/dz`, velocity, displacement, uncertainty, impulse, burden velocity, contributing event, facies, classes, grade, tonnes and carats.
- Per event: nominal/actual firing time, timing error, charge, VOD, pressure proxy, energy, gas-decay time, stemming effectiveness, burden velocity and released voxel count.
- Ore control: recovery, loss, dilution, feed grade, carat recovery and source/destination mixing.
- Loader scale: recovery and dilution at the configured minimum mining unit.
- Remap: collision count, occupied cells and mass/carat conservation flags.

Every export is labelled `Synthetic Demonstration / Uncalibrated — Planning Only`.

## Mine-data onboarding

Production calibration requires final as-drilled/as-charged deck records, actual timing, bench/free-face geometry, pre/post drone or LiDAR surfaces, the grade-control block model, measured movement vectors, dig polygons, truck destinations and reconciliation. Validation must split by blast and report vector MAE/RMSE/bias, surface error, ore-control reconciliation, uncertainty calibration and drift.
