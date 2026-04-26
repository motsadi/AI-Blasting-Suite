# Delay Timing Algorithms

The module uses transparent deterministic timing rules. It does not use machine learning for delay assignment.

## Row Detection

Rows are inferred by binning coordinates using the configured row tolerance. By default, the module groups by `Y` for left/right directions. For top/bottom directions it groups by `X`. Holes inside each row are sorted along the other axis.

## Row-By-Row

Rows fire in order. Holes inside each row fire by column order.

```text
delayMs = startDelayMs + rowIndex * rowDelayMs + columnIndex * inRowDelayMs
```

## Chevron

Rows are detected first. Inside each row, holes closer to the selected centre point fire earlier. This produces a centre-out V or chevron-like progression.

```text
delayMs = startDelayMs + rowIndex * rowDelayMs + lateralOrder * inRowDelayMs
```

## V-Cut

The selected hole is used as the apex when available. Holes are sorted by distance from the apex, with a width factor controlling how strongly the V shape spreads outward.

## Box-Cut / Centre-Out

The selected hole or project centre is used as the centre. Holes are grouped into distance rings. Inner rings fire first.

```text
delayMs = startDelayMs + ringIndex * rowDelayMs + withinRingOrder * inRowDelayMs
```

## Directional From Point

The selected hole or project centre is used as the initiation point. Holes closer to that point fire earlier.

```text
delayMs = startDelayMs + order * inRowDelayMs
```

## Directional From Line

If two holes are selected, they define the initiation line. Otherwise a default line is generated across the bottom of the layout. Holes closest to the line fire earlier, then sort by along-line position.

## Rounding

When rounding is enabled:

```text
roundedDelay = round(delayMs / delayIncrementMs) * delayIncrementMs
```
