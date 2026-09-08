# GeoMotion 3D Test Plan

Use this checklist before deploying changes to the GeoMotion 3D module.

## GeoMotion demonstration

- Click the main-app GeoMotion navigation entry and confirm it opens or focuses the authenticated `?view=geomotion` standalone window; the standalone viewport must contain no application header or sidebar.
- Block `window.open` and confirm the launch page announces the popup failure and retains retry plus normal-tab fallbacks.
- Open GeoMotion without importing data and confirm the initial `Whole pit overview` visibly contains the complete crest, seven descending benches, highwall faces, and depressed floor—not a mound or raised ore body.
- Confirm the tie-up and movement cells remain a small bounded section on Bench 680 / East 04 while the complete pit stays dominant.
- Use `Focus active blast section`; confirm holes, tie lines and ordered delays are readable and the whole-pit locator identifies the section. Use `Return to whole pit` and confirm the fitted crest returns.
- Replace the demonstration with an uploaded blast CSV and confirm only the active tie-up changes; the whole-pit context remains.
- Test native Fullscreen API entry/exit and Escape, then disable/reject the API and confirm the viewport fallback exits safely with Escape.
- Orbit, pan, wheel-zoom, use the accessible +/- controls, and reset/fit the whole pit.
- Play, pause, scrub, and reset the firing timeline; confirm fired/current/queued state changes are limited to the active bench.
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
