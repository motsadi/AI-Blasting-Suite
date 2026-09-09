# GeoMotion 3D Test Plan

Use this checklist before deploying changes to the GeoMotion 3D module.

## GeoMotion demonstration

- Click the main-app GeoMotion navigation entry and confirm it opens or focuses the authenticated `?view=geomotion` standalone window; the standalone viewport must contain no application header or sidebar.
- Block `window.open` and confirm the launch page announces the popup failure and retains retry plus normal-tab fallbacks.
- Open GeoMotion without importing data and confirm the viewer is an empty block-model viewport with no bench floor, highwall, free face, footprint, north arrow, whole-pit mesh, distant terrain or mine locator.
- After running the demonstration, confirm only 1 m³ cubes are drawn, ore is gold, waste is stone, cube edges and seams are visible, and Fit frames the whole model.
- Confirm holes and coordinates stay in imported plan space and are not projected onto a synthetic pit wedge.
- Replace the demonstration with an uploaded blast CSV and confirm camera fit and model dimensions adapt to that tie-up.
- Test native Fullscreen API entry/exit and Escape, then disable/reject the API and confirm the viewport fallback exits safely with Escape.
- Orbit, pan, wheel-zoom, use the accessible +/- controls, and reset/fit the whole block model.
- Play, pause, scrub, and reset the firing timeline; confirm the delay-order HUD and bulk-rock cubes follow the delay sequence.
- Confirm displacement colours use the labelled zero-to-P95 scale and optional movement vectors run from real source to destination coordinates.
- Confirm the UI states that displacement is uncalibrated bulk-rock movement—not flyrock, damage or an exclusion zone.
- Check desktop/mobile layouts and both light/dark application themes.
- Load the built-in delay-bearing 182-hole diamond demonstration and confirm every physics cell is exactly 1 m³ (1 m × 1 m × 1 m).
- Import `Hole_data_v1.csv`; confirm cumulative times are preserved, normalized to zero and its invalid depth/charge rows are explicitly excluded.
- Confirm a tie-up without Delay, with missing Delay or duplicate firing times cannot run.
- Run physics mode twice with the same seed and confirm identical results.
- Run the uncertainty realization and confirm electronic scatter changes actual event times reproducibly.
- Verify S135B density/RWS/VOD assumptions, 400 g Pentolite default and 127/165/250 mm linear-loading checks.
- Inspect event pressure proxy, impulse, burden velocity and evolving release history.
- Confirm tonnes and contained-carats attributes move with each block and mass-balance error remains zero.
- Confirm conservative remapping reports unique occupied cells and collision settlement.
- Confirm ore recovery plus ore loss is approximately 100% for source ore.
- Confirm loader/MMU recovery and dilution are reported separately.
- Switch among in-situ, movement timeline, and post-blast views.
- Orbit, zoom, clip sections, change camera and Z exaggeration, join/separate voxels, toggle vectors, and colour by all overlays.
- Export LOD movement CSV, result JSON and authoritative full-resolution CSV.gz; confirm synthetic warnings.
- Confirm the browser fallback identifies itself as a coarse preview and never claims an authoritative 1 m³-cell result.
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
