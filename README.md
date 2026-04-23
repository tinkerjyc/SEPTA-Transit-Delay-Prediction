# SEPTA Transit Delay Prediction — Deliverable 2
**CISC 520 Data Mining and Engineering · Harrisburg University**  
Team: Yicheng Jiang · Caleb Ehireme Okoh-Aih · Boxuan Chen

---

## Overview

This project collects real-time SEPTA transit data, computes per-vehicle delays by
joining against the GTFS static schedule, and produces 22 visualisation figures
covering the full network and three modal deep-dives (Regional Rail, Bus,
Subway & Trolley).

```
project/
├── collect_septa.py     — Step 1: poll SEPTA real-time API → septa_realtime_raw.csv
├── download_gtfs.py     — Step 2: download SEPTA GTFS static schedule (one-time)
├── analysis.py          — Step 3: compute delays, generate 22 figures + stats_report.txt
└── README.md            — This file
```

---

## Quick Start

### Install dependencies (one-time)

```bash
pip install pandas numpy matplotlib seaborn scipy requests gtfs-realtime-bindings protobuf
```

### Step 1 — Start collecting real-time data

```bash
mkdir E:\SEPTA_data
cd E:\SEPTA_data
python collect_septa.py
```

Leave it running. It writes to `E:\SEPTA_data\septa_realtime_raw.csv` every 90 seconds.
Safe to stop and restart — the file is append-only, no data is lost.

### Step 2 — Download the GTFS static schedule (one-time, ~60 seconds)

Open a second terminal while the collector runs:

```bash
python download_gtfs.py
```

This downloads `stop_times.txt` (~200 MB) into `E:\SEPTA_data\gtfs\`.
Required for bus and trolley delay computation. Only needs to run once.

### Step 3 — Generate figures and statistics

```bash
python analysis.py
```

Reads `E:\SEPTA_data\septa_realtime_raw.csv`, joins against the GTFS schedule,
and writes 22 PNG figures + `stats_report.txt` to the same folder as `analysis.py`.

---

## File Details

### `collect_septa.py`

Polls two SEPTA GTFS-RT TripUpdate endpoints every 90 seconds:

| Feed | URL |
|------|-----|
| Bus + Trolley | `https://www3.septa.org/gtfsrt/septa-pa-us/Trip/rtTripUpdates.pb` |
| Regional Rail | `https://www3.septa.org/gtfsrt/septarail-pa-us/Trip/rtTripUpdates.pb` |

Each poll extracts one row per active trip — the vehicle's next stop and its
predicted arrival time — and appends it to `septa_realtime_raw.csv`.

**Key setting:**

```python
NEXT_STOP_ONLY = True   # keeps only the first stop-time-update per trip
                         # set False to capture all future stops (warning: ~8 GB/48h)
```

**Output CSV columns:**

| Column | Description |
|--------|-------------|
| `collected_at` | ISO-8601 UTC timestamp of the poll |
| `feed_type` | `bus_trolley` or `rail` |
| `trip_id` | GTFS trip identifier |
| `route_id` | Route (e.g. `44`, `LAN`, `T1`) |
| `direction_id` | `0` = inbound, `1` = outbound |
| `start_date` | Service date `YYYYMMDD` (ET date, derived from poll time when absent) |
| `stop_id` | GTFS stop identifier |
| `stop_sequence` | Position of this stop along the route |
| `arrival_delay_sec` | Explicit delay in seconds — populated by rail feed only |
| `departure_delay_sec` | Explicit departure delay — populated by rail feed only |
| `arrival_time_unix` | Predicted absolute arrival (Unix seconds) — populated by bus feed |
| `departure_time_unix` | Predicted absolute departure (Unix seconds) |
| `schedule_rel` | `SCHEDULED` / `SKIPPED` / `NO_DATA` |
| `vehicle_id` | Vehicle identifier |

**Expected file sizes:**

| Duration | Approx. rows | Approx. size |
|----------|-------------|--------------|
| 2 hours | ~80,000 | ~15 MB |
| 12 hours | ~480,000 | ~90 MB |
| 48 hours | ~1,900,000 | ~360 MB |

**Note on MFL / BSL (subway):**
SEPTA's subway lines publish only vehicle position feeds, not trip update feeds.
As a result, no arrival delay data is available for MFL or BSL through this pipeline.
This is a documented SEPTA API limitation, not a code issue.

---

### `download_gtfs.py`

Downloads `gtfs_public.zip` from SEPTA's GitHub releases and extracts
`stop_times.txt` into `E:\SEPTA_data\gtfs\`.

`stop_times.txt` contains the scheduled arrival time for every trip at every stop.
`analysis.py` uses it to compute bus/trolley delays as:

```
delay = arrival_time_unix − scheduled_arrival_unix
```

where `scheduled_arrival_unix = ET_midnight + scheduled_arrival_sec`.

The file is ~200 MB uncompressed and only needs to be downloaded once.
Re-run if SEPTA releases a new schedule (roughly every few months).

```bash
python download_gtfs.py
```

Expected output:
```
Downloading GTFS static feed ...
✓ stop_times.txt saved to: E:\SEPTA_data\gtfs\stop_times.txt
  File size: 198 MB
✓ GTFS static feed ready.
```

---

### `analysis.py`

Loads the collected CSV, computes delays for all modes, and generates
22 figures + a statistics report. Auto-detects real vs synthetic data.

**Two modes — no code change needed:**

| Mode | When | What |
|------|------|------|
| **Real** | `septa_realtime_raw.csv` exists with > 100 rows | Loads and processes real SEPTA data |
| **Synthetic** | CSV absent or too small | Generates a calibrated fallback dataset |

**Delay computation by mode:**

| Mode | Method | Source field |
|------|--------|-------------|
| Regional Rail | Explicit protobuf delay field | `arrival_delay_sec` (seconds) |
| Bus / Trolley | GTFS static join | `arrival_time_unix − scheduled_arrival_unix` |
| Subway (MFL/BSL) | Not available | SEPTA API limitation — no trip updates published |

**GTFS join logic:**

Because SEPTA's real-time `trip_id` values (e.g. `734808`) do not match the
GTFS static `trip_id` values (e.g. `656954`) — they use different internal
numbering systems — the join is performed on `stop_id` only. For each RT row,
the closest scheduled trip serving that stop within ±2 hours of the actual
arrival time is selected as the match.

**Deduplication:**

Rail and bus rows are deduplicated differently because the rail feed reports
trip-level delays with no `stop_id`:

- **Rail:** one row per `(trip_id, 2-min poll window)`
- **Bus/Trolley:** one row per `(trip_id, stop_id, 2-min poll window)`

**Output figures (22 total):**

| Set | Figures | Description |
|-----|---------|-------------|
| Global | `fig1`–`fig5`, `fig7`, `fig8` | Full network: histogram, hourly line, mode box plots, heatmap, correlation matrix, Q-Q plot, scatter plot |
| Regional Rail | `fig_rail_1`–`5` | Mean delay by line (95% CI), hour×line heatmap, inbound vs outbound, box plots by line, hourly profiles per line |
| Bus | `fig_bus_1`–`5` | Mean delay by route, weekday vs weekend hourly, top-10 box plots, hour×day heatmap, on-time rate by route |
| Subway & Trolley | `fig_subway_1`–`5` | Box plots by route, mean delay bar, subway vs trolley hourly, weekday vs weekend, hour×route heatmap |

**Output stats (`stats_report.txt`):**

Saved to the same folder as `analysis.py`. Contains full descriptive statistics,
class balance, per-mode breakdowns, Pearson correlations, and Kruskal-Wallis test.
Upload this file (3–5 KB) for remote review instead of the full CSV.

---

## Data Collection Notes

### Why 48 hours?

The project deadline constrained collection to 48 hours. This window captures
one full weekday cycle including overnight minimal service, AM peak, midday,
and the start of PM peak. A production deployment would collect continuously
for weeks or months, at which point the pipeline would need to migrate from
pandas (CSV) to a distributed framework (Apache Spark or Dask) with columnar
storage (Parquet).

### Overnight negative delays

Buses frequently show negative delays (arriving early) during overnight hours
(midnight–6 AM). This is real and expected: SEPTA's owl bus schedules use
conservative time buffers, and buses run significantly faster on empty roads.
Mean delays shift positive once AM peak traffic builds.

### Eastern Time handling

SEPTA schedules run on Eastern Time (ET). All timestamp conversions in
`analysis.py` apply a UTC−4 offset (EDT, April–October) before computing
time-of-day comparisons against GTFS scheduled times. This is handled via
manual offset subtraction rather than `zoneinfo` to ensure consistent behaviour
across Python versions and operating systems.

---

## Dependencies

| Package | Install |
|---------|---------|
| pandas | `pip install pandas` |
| numpy | `pip install numpy` |
| matplotlib | `pip install matplotlib` |
| seaborn | `pip install seaborn` |
| scipy | `pip install scipy` |
| requests | `pip install requests` |
| gtfs-realtime-bindings | `pip install gtfs-realtime-bindings` |
| protobuf | `pip install protobuf` |

Python 3.8+ required.

---

## Troubleshooting

**`⚠ GTFS static feed not found`**  
→ Run `python download_gtfs.py` first. Bus/trolley delays cannot be computed
without `E:\SEPTA_data\gtfs\stop_times.txt`.

**`GTFS join: 0/X matched`**  
→ Check that `stop_id` values in your CSV match SEPTA's GTFS format (plain integers
like `18013`, not `18013.0`). The script strips `.0` suffixes automatically but
verify the raw CSV with:
```bash
python -c "import pandas as pd; print(pd.read_csv(r'E:\SEPTA_data\septa_realtime_raw.csv', nrows=3)[['stop_id','route_id']])"
```

**`source = synthetic` when real CSV exists**  
→ Check that `E:\SEPTA_data\septa_realtime_raw.csv` has more than 100 rows.
The script falls back to synthetic if the file is too small.

**MFL / BSL missing from subway figures**  
→ This is expected. SEPTA does not publish GTFS-RT TripUpdate feeds for subway
lines. Only vehicle position data is available, which cannot be used to compute
schedule adherence without additional stop-matching logic.

**Collector shows low row counts (< 200/poll)**  
→ Normal during overnight hours (midnight–5 AM) when only a handful of trains
and owl buses are operating. Row counts recover to 600–800/poll during AM peak.
