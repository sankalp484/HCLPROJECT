# Data Card — Multi-Plant Order Allocation

## Name and version
Synthetic data pack v1.1 — 2026-10-11

## How it was made
Generated synthetically using Python scripts with seed=42.
No real company or personal data used.

## What it contains
| Table | Rows | Date Range |
|-------|------|------------|
| orders.csv | 150 | production weeks 2026-10-12 to 2026-11-30; due 2026-10-24 to 2027-01-30 |
| plants.csv | 80 | 10 plant/product pairs (A: P1-P3, B: P1/P2/P4, C: P1/P3/P4/P5) x 8 weeks, 2026-10-12 to 2026-11-30 |
| transport_duties.csv | 36 | 3 plants x 12 destinations, no dates |
| capacity_events.csv | 3 | 2026-10-19 to 2026-11-29 |

Planning start: 2026-10-12 (first plant week).

## Units
- `weekly_capacity`: units per week (`capacity_unit` column), **shared by all products a plant makes that week** (the value repeats on each product row)
- `unit_cost`, `energy_price_per_kwh`, `transport_cost_per_unit`: INR (`cost_currency` column)
- `carbon_kg_per_kwh`: kg CO2e per kWh (`carbon_unit` column); `carbon_kg_per_unit`: kg CO2e per unit
- `duty_pct`, `capacity_loss_pct`: fractions (0.25 = 25%)

Columns follow Annex B exactly (`orders.due`, v1.1 renamed it from `due_date`). Extra columns: `orders.production_week`, `plants.week_start`,
`plants.energy_price_per_kwh` and the unit columns. The loader treats these as optional.

## Traps and edge cases
Counts come from `scripts/validate_data.py` (arrival = production week start + 6 days + transit days).
- Cheapest plant arrives late: 35 orders, e.g. ORD-007 (Plant A 14 days late, Plant B on time)
- No plant can deliver on time: 5 orders (ORD-054, ORD-099, ORD-102, ORD-110, ORD-113)
- Only one plant can make the order: 18 orders, all P5 (only Plant C makes P5)
- Chicago duty: A=25%, B=6%, C=0%, which changes which plant is cheapest
- Plant C: lowest carbon (0.17-0.31 kg/kWh) but only 390-405 units/week
- E001: Plant B loses 30% for two weeks (2026-11-02 to 2026-11-15), the late data-drop scenario
- E002: Plant C loses 85% in week 2026-11-23 (390 -> 58 units), but that week's P5 orders need 150 units -> some P5 orders infeasible
- E003: Plant A loses 10% in week 2026-10-19 (810 -> 729 units)
- Carbon intensity changes weekly (A 0.80-0.85, B 0.55-0.69, C 0.17-0.31)

## Checks run
All checks in `scripts/validate_data.py` pass, including row counts against this card.

## Known limits
- Transit times are fixed; no real-world variability
- No currency fluctuation; duty rates constant for the period
- No supplier or material constraints

## Privacy
No real company or personal data included.
