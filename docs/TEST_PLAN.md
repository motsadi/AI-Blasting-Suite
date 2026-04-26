# Delay Module Test Plan

Use this checklist before deploying changes to the Delay Design & Simulation module.

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
