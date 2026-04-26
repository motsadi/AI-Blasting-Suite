# Delay Design User Guide

## Start A Project

Open AI Blasting Suite and select `Delay Design & Simulation`. Enter a project name in Project Setup.

## Import Hole Data

Click the CSV import control and choose `680-665QS32-33 Tie-Up.csv` or another CSV with at least `X` and `Y` columns. Confirm the column mapping if the automatic mapping is not correct.

## Validate Data

Review the Validation panel after import. Errors should be fixed before assigning delays. Warnings identify items such as blank IDs, duplicate IDs, missing depth, missing charge, or possible outliers.

## Visualise The Pattern

The Blast Layout panel shows every imported hole in equal-aspect plan view. Use the display toggles to show hole IDs, firing order numbers, fired holes, unfired holes, and active wavefronts.

## Assign Timing

Choose a timing pattern:

- Row-by-row
- Chevron
- V-cut
- Box-cut / centre-out
- Directional from point
- Directional from line
- Manual timing

Adjust start delay, in-row delay, row-to-row delay, row tolerance, rounding, and direction. Click `Assign delays`.

For centre-based patterns, click a hole first to use it as the centre or apex. For line-directional timing, select two holes to define the line.

## Run Simulation

Use Play, Pause, Reset, Step next, speed, and the timeline slider. Current holes are highlighted while fired and unfired holes use different visual states.

## Manually Edit Delays

Click a hole and use the Selected Hole panel to edit or clear its delay.

## Export

After assigning delays, export:

- Delay assignment CSV
- Project JSON
- Printable planning report

All exports are Planning/Simulation Draft outputs and must be reviewed by qualified blasting personnel before any field use.
