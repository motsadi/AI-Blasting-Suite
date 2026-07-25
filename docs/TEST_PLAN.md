# GeoMotion 3D Test Plan

Use this checklist before deploying changes to the GeoMotion 3D module.

## GeoMotion demonstration

- Load the built-in 182-hole diamond demonstration and confirm the stated 6 m burden, 7 m spacing, 250 mm diameter, and diamond defaults.
- Import the supplied charged-hole tie-up and confirm missing IDs, duplicate IDs, and near-overlapping collars are surfaced rather than silently removed.
- Run physics mode twice with the same seed and confirm identical results.
- Run hybrid mode and confirm the model is identified as a synthetic random-forest residual, not a calibrated mine model.
- Confirm tonnes and contained-carats attributes move with each block and mass-balance error remains zero.
- Confirm ore recovery plus ore loss is approximately 100% for source ore.
- Switch among in-situ, movement timeline, and post-blast views.
- Orbit, zoom, toggle vectors, and colour by classification, facies, grade, displacement, and uncertainty.
- Export movement CSV and result JSON; confirm both carry `Synthetic Demonstration / Uncalibrated — Planning Only`.
- Confirm failed API calls and malformed inputs produce visible errors.
- Confirm the module contains no arming, firing, detonator programming, or hardware controls.

## Legacy timing regression

- Import a valid CSV with `Hole ID`, `Depth`, `Charge`, `X`, `Y`, and `Z`.
- Import a CSV with blank or `N/A` Hole IDs and confirm generated IDs such as `H001`.
- Import a CSV with duplicate Hole IDs and confirm warnings are shown.
- Import a CSV with missing or non-numeric `X` or `Y` and confirm errors are shown.
- Generate row-by-row timing and confirm delay values follow row and column order.
- Generate chevron timing from a selected centre hole.
- Generate V-cut timing from a selected apex hole.
- Generate box-cut timing from a selected centre hole.
- Generate point-directional timing from a selected hole.
- Generate line-directional timing after selecting two holes.
- Manually edit a selected hole delay.
- Reset timing and confirm delays are cleared.
- Run Play, Pause, Reset, Step next, speed, and timeline simulation controls.
- Toggle labels, firing order numbers, fired holes, unfired holes, and active wavefront.
- Export CSV after assigning delays.
- Export JSON project data.
- Export/open printable report.
- Confirm all exported outputs are labelled as Planning/Simulation Draft.
