# Delay Module Data Format

The Delay Design & Simulation module accepts CSV files with blast-hole rows.

Required columns:

- `X`
- `Y`

Optional columns:

- `Hole ID`
- `Z`
- `Depth`
- `Charge`

Example:

```csv
Hole ID,Depth,Charge,X,Y,Z
N/A,14.116,611.92948,-5399.442,4624.935,678.116
N1,14.454,616.0270495,-5363.884,4603.971,678.454
N10,14.758,619.5834305,-5418.444,4633.164,678.758
O4,14.886,624.8793458,-5381.112,4620.648,678.886
```

## Missing Hole IDs

If `Hole ID` is blank, missing, `N/A`, `nan`, `none`, or `null`, the module generates a temporary ID such as `H001`. The original imported value is preserved in `Original Hole ID` for export and review.

## Duplicate IDs

Duplicate IDs are kept but flagged as validation warnings. Review duplicates before exporting or using the timing design for planning review.

## Column Mapping

The module automatically maps common names such as `Hole ID`, `X`, `Y`, `Z`, `Depth`, and `Charge`. If mapping is wrong or incomplete, select the correct columns in the Project Setup panel and apply the mapping.
