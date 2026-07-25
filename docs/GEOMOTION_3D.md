# GeoMotion 3D Engine

GeoMotion 3D is an independent, physics-informed blast-movement demonstration. It transforms a charged-hole tie-up and a synthetic pre-blast diamond block model into a mass-conserving post-blast material model.

## Status and safety

All current geology, measured movement, calibration, grade, recovery, dilution, and uncertainty outputs are synthetic. They are intended for software demonstration and workflow design only. They are not validated predictions for Orapa or any other mine and must not be used for field execution, dig-limit control, resource reporting, or production decisions.

The module does not arm, program, or communicate with detonators. The repository-wide requirements in `SAFETY_SCOPE.md` remain in force.

## Input contract

The tie-up accepts:

| Field | Requirement | Notes |
|---|---|---|
| Hole ID | Recommended | Missing IDs receive generated identifiers |
| X, Y | Required | Coordinates in a consistent projected metre system |
| Z | Recommended | Collar RL |
| Depth | Recommended | Metres |
| Charge | Recommended | Kilograms |
| Delay | Optional | Milliseconds; an open V-style demonstration sequence is generated when absent |

The diamond demonstration defaults are based on the supplied 680-665QS32-33 context: 250 mm holes, 6 m burden, 7 m spacing, 5.02 m stemming, 1 m subdrill, 2.35 t/m³ rock density, 0.92 kg/m³ powder factor, and an open V-chevron concept.

Input QA reports duplicate IDs, near-overlapping collars, imputed fields, nearest-hole spacing, and an inferred floor RL. It never silently deletes an uploaded record.

## Synthetic geology

The engine creates a seeded 3D block model around the blast footprint. Each cell carries:

- Source and destination coordinates
- Kimberlite facies (`VK`, `SVK_M1`, contact, or waste)
- Density and tonnes
- Synthetic grade in carats per hundred tonnes (cpht)
- Ore/waste classification
- Contained synthetic carats

The seed makes a demonstration repeatable. Neither facies nor grade is inferred from the tie-up.

## Movement model

The physics baseline combines effective explosive energy, distance attenuation, burden and spacing, free-face direction, timing relief, confinement, depth, heave, throw, and swell. Overlapping hole influence produces one 3D movement vector per cell. Movement is bounded, and the source cell's tonnes, grade, facies, and contained carats travel with that cell.

Hybrid mode applies a small random-forest residual trained on seeded synthetic parameter sweeps. This demonstrates the production calibration interface, but does not add mine accuracy. When measured mine data become available, the synthetic residual must be replaced with a site model trained on measured `dx`, `dy`, `dz`, post-blast surfaces, and reconciliation outcomes. Validation splits must be by blast, not random cells from the same blast.

## Output definitions

- **Ore recovery:** in-situ ore tonnes remaining inside the synthetic post-blast ore destination divided by in-situ ore tonnes.
- **Ore loss:** in-situ ore tonnes moved outside the synthetic ore destination divided by in-situ ore tonnes.
- **Dilution:** source-waste tonnes entering the synthetic ore destination divided by total post-blast ore-stream tonnes.
- **Carat recovery:** contained synthetic carats retained in the ore destination divided by in-situ contained synthetic carats.

Exports include per-cell vectors, uncertainty, facies, source/destination classes, grade, tonnes, and contained carats. Every export is labelled `Synthetic Demonstration / Uncalibrated — Planning Only`.

## Mine-data onboarding

Production calibration will require, per blast:

1. Final as-drilled and as-charged hole/deck records with actual timing.
2. Bench geometry, free faces, pre-blast surface, and post-blast drone/LiDAR surface.
3. Grade-control block model with facies, density, grade, classification rules, and coordinate reference system.
4. Measured movement vectors from monitors or surveyed markers where available.
5. Final dig polygons, truck destinations, plant feed, and reconciliation.

The production model should report spatial cross-validation, vector MAE/RMSE/bias, surface error, mass balance, recovery/dilution reconciliation, uncertainty calibration, data drift, and the domain over which predictions are supported.
