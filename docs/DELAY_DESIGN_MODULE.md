# Delay Design & Simulation Module

The Delay Design & Simulation module replaces the previous Delay Prediction tab inside AI Blasting Suite. It is an integrated planning tool for importing blast-hole coordinates, assigning timing delays, reviewing a firing simulation, validating the design, and exporting planning draft outputs.

It is for blasting engineers, students, trainers, and reviewers who need a transparent timing design workspace inside the existing software. It does not arm, fire, program, connect to, or communicate with detonator hardware.

## Run The App

From `frontend/`:

```bash
npm install
npm run dev
```

The module is available from the app navigation under `Delay Design & Simulation`.

## Basic Workflow

1. Open the Delay Design & Simulation tab.
2. Import a CSV file such as `680-665QS32-33 Tie-Up.csv`.
3. Confirm the column mapping for Hole ID, X, Y, Z, Depth, and Charge.
4. Validate the imported holes.
5. Choose a timing pattern and settings.
6. Click `Assign delays`.
7. Review the 2D layout, selected hole details, validation messages, and analysis indicators.
8. Run the simulation with Play, Pause, Reset, Step next, speed, and timeline controls.
9. Export the delay assignment CSV, project JSON, or printable planning report.

## CSV Export

After delays have been assigned, export a CSV named like:

```text
<project-name>_delay_assignment_planning_draft.csv
```

The export includes every imported hole, generated temporary IDs, original imported ID values, delay values, firing order, row/column indices, timing group, warnings, and a Planning/Simulation Draft note.
