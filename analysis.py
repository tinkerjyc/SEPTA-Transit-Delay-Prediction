"""
analysis.py
===========
SEPTA Transit Delay Prediction — Deliverable 2
CISC 520 Data Mining and Engineering, Harrisburg University

GTFS STATIC JOIN (required for real bus/trolley delay computation)
------------------------------------------------------------------
Bus/trolley rows in the real-time feed only report arrival_time_unix
(absolute predicted arrival), NOT a delay field. To compute actual
delay we must compare arrival_time_unix against the scheduled arrival
from the GTFS static stop_times.txt file.

Run download_gtfs.py ONCE first to fetch the GTFS static feed:
    python download_gtfs.py
This saves  E:\\SEPTA_data\\gtfs\\google_bus.zip  (extracted automatically).

Without the GTFS static feed, bus/trolley delays cannot be computed and
those modes will be excluded from the analysis with a clear warning.

DEPENDENCIES
------------
    pip install pandas numpy matplotlib seaborn scipy

USAGE
-----
    python analysis.py
"""

import io
import os
import warnings
import zipfile
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats

warnings.filterwarnings("ignore")


# ═════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═════════════════════════════════════════════════════════════════════════

RANDOM_SEED     = 42
N_SYNTHETIC     = 12_000
OUT_DIR         = os.path.dirname(os.path.abspath(__file__))

# Real-time CSV — check E: portal drive first, then script folder
_PORTAL_CSV = r"E:\SEPTA_data\septa_realtime_raw.csv"
_LOCAL_CSV  = os.path.join(OUT_DIR, "septa_realtime_raw.csv")
REAL_CSV_PATH = _PORTAL_CSV if os.path.exists(_PORTAL_CSV) else _LOCAL_CSV

# GTFS static feed — bus zip extracted by download_gtfs.py
_PORTAL_GTFS = r"E:\SEPTA_data\gtfs"
_LOCAL_GTFS  = os.path.join(OUT_DIR, "gtfs")
GTFS_DIR = _PORTAL_GTFS if os.path.exists(_PORTAL_GTFS) else _LOCAL_GTFS

DELAY_MIN_CLIP  = -30.0   # minutes — allow up to 30 min early
DELAY_MAX_CLIP  =  90.0   # minutes — cap at 90 min late
DELAY_THRESHOLD =   5.0   # minutes — is_delayed = 1 if delay >= this

# Colours
NAVY  = "#1F3864"
BLUE  = "#2E75B6"
LBLUE = "#9DC3E6"
GOLD  = "#C9A84C"
RED   = "#C0392B"
GREEN = "#27AE60"
GRAY  = "#95A5A6"
PURP  = "#8E44AD"

MODE_COLORS = {
    "Regional Rail": NAVY,
    "Bus":           BLUE,
    "Trolley":       GOLD,
    "Subway":        GREEN,
}

plt.rcParams.update({
    "font.family":       "DejaVu Sans",
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.3,
    "grid.linestyle":    "--",
})


# ═════════════════════════════════════════════════════════════════════════
# ROUTE → MODE CLASSIFICATION
# ═════════════════════════════════════════════════════════════════════════

_RAIL_CODES = {
    "AIR","CHE","CHW","CYN","FOX","LAN","MED",
    "NOR","PAO","TRE","WAR","WIL","WTR",
}
_RAIL_FULL_NAMES = {
    "AIR":"Airport",           "CHE":"Chestnut Hill East",
    "CHW":"Chestnut Hill W",   "CYN":"Cynwyd",
    "FOX":"Fox Chase",         "LAN":"Lansdale/Doylestown",
    "MED":"Media/Wawa",        "NOR":"Norristown",
    "PAO":"Paoli/Thorndale",   "TRE":"Trenton",
    "WAR":"Warminster",        "WIL":"Wilmington/Newark",
    "WTR":"West Trenton",
}

def classify_mode(route_id: str) -> str:
    r = str(route_id).strip().upper()
    if r in _RAIL_CODES:                          return "Regional Rail"
    if r in {"MFL","BSL","M2","M3","M4"}:         return "Subway"
    if r.startswith("T") or r in {"T BUS","T5 BUS"}: return "Trolley"
    return "Bus"

def rail_label(route_id: str) -> str:
    return _RAIL_FULL_NAMES.get(str(route_id).strip().upper(), str(route_id))


# ═════════════════════════════════════════════════════════════════════════
# GTFS STATIC SCHEDULE LOADER
# ═════════════════════════════════════════════════════════════════════════

def _gtfs_time_to_seconds(t: str) -> int:
    """Convert 'HH:MM:SS' (or '25:30:00' for post-midnight) to seconds."""
    try:
        parts = str(t).strip().split(":")
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
    except Exception:
        return -1


def load_gtfs_stop_times() -> pd.DataFrame | None:
    """
    Load stop_times.txt from the GTFS static bus feed.
    Returns a DataFrame with columns:
        trip_id, stop_id, stop_sequence, scheduled_arrival_sec
    or None if the GTFS feed is not available.
    """
    # Look for extracted stop_times.txt or the zip
    candidates = [
        os.path.join(GTFS_DIR, "stop_times.txt"),                # extracted
        os.path.join(GTFS_DIR, "google_bus", "stop_times.txt"),  # sub-folder
    ]
    zip_candidates = [
        os.path.join(GTFS_DIR, "google_bus.zip"),
        os.path.join(GTFS_DIR, "gtfs_public.zip"),
    ]

    stop_times_path = None
    for p in candidates:
        if os.path.exists(p):
            stop_times_path = p
            break

    # Try extracting from zip if not found
    if stop_times_path is None:
        for zp in zip_candidates:
            if os.path.exists(zp):
                try:
                    print(f"  Extracting stop_times.txt from {zp} ...")
                    with zipfile.ZipFile(zp, "r") as z:
                        names = z.namelist()
                        # Handle nested zips (gtfs_public.zip contains google_bus.zip)
                        if "google_bus.zip" in names:
                            inner = z.read("google_bus.zip")
                            import io as _io
                            with zipfile.ZipFile(_io.BytesIO(inner)) as iz:
                                iz.extract("stop_times.txt", GTFS_DIR)
                        elif "stop_times.txt" in names:
                            z.extract("stop_times.txt", GTFS_DIR)
                        stop_times_path = os.path.join(GTFS_DIR, "stop_times.txt")
                        break
                except Exception as e:
                    print(f"  Warning: could not extract from {zp}: {e}")

    if stop_times_path is None or not os.path.exists(stop_times_path):
        print("\n  ⚠  GTFS static feed not found.")
        print("     Run download_gtfs.py first to enable bus/trolley delay computation.")
        print(f"     Expected location: {GTFS_DIR}\n")
        return None

    print(f"  Loading GTFS stop_times from: {stop_times_path}")
    st = pd.read_csv(stop_times_path, usecols=["trip_id","arrival_time",
                                                "stop_id","stop_sequence"],
                     dtype=str, low_memory=False)
    st["scheduled_arrival_sec"] = st["arrival_time"].apply(_gtfs_time_to_seconds)
    st = st[st["scheduled_arrival_sec"] >= 0]
    st["stop_sequence"] = pd.to_numeric(st["stop_sequence"], errors="coerce")
    print(f"  GTFS stop_times loaded: {len(st):,} rows")
    return st[["trip_id","stop_id","stop_sequence","scheduled_arrival_sec"]]


# ═════════════════════════════════════════════════════════════════════════
# REAL DATA LOADER
# ═════════════════════════════════════════════════════════════════════════

def load_real_dataset(path: str = REAL_CSV_PATH) -> pd.DataFrame:
    """
    Load and clean the CSV produced by collect_septa.py.

    Delay computation:
      Rail    → explicit arrival_delay_sec  (reliable, from protobuf delay field)
      Bus/T   → arrival_time_unix − scheduled_arrival_time (from GTFS static join)
                If GTFS unavailable → bus/trolley rows excluded with warning
    """
    print(f"  Loading: {path}")
    raw = pd.read_csv(path, low_memory=False)
    print(f"  Raw rows: {len(raw):,}")

    # Parse timestamps
    raw["collected_at"] = pd.to_datetime(raw["collected_at"], utc=True, errors="coerce")
    raw = raw.dropna(subset=["collected_at"]).copy()

    # Numeric columns
    for col in ["arrival_delay_sec","departure_delay_sec",
                "arrival_time_unix","departure_time_unix"]:
        raw[col] = pd.to_numeric(raw[col], errors="coerce")

    # Route / mode early (needed for split logic below)
    # Clean stop_id and trip_id — pandas reads integer-looking floats as "18013.0"
    raw["stop_id"]  = (raw["stop_id"].astype(str).str.strip()
                        .str.replace(r"\.0$", "", regex=True))
    raw["trip_id"]  = (raw["trip_id"].astype(str).str.strip()
                        .str.replace(r"\.0$", "", regex=True))
    raw["route"]    = raw["route_id"].fillna("Unknown").astype(str).str.strip()
    raw["mode"]     = raw["route"].apply(classify_mode)

    # ── Rail: use explicit delay field ──────────────────────────────────
    rail_mask = raw["mode"] == "Regional Rail"
    raw.loc[rail_mask, "delay_sec"] = (
        raw.loc[rail_mask, "arrival_delay_sec"]
        .fillna(raw.loc[rail_mask, "departure_delay_sec"])
    )
    n_rail_explicit = int(rail_mask.sum())

    # ── Bus/Trolley/Subway: GTFS static join ────────────────────────────
    surface_mask = ~rail_mask
    n_surface    = int(surface_mask.sum())
    n_surface_ok = 0

    # Diagnostic: show sample of raw surface rows
    if n_surface > 0:
        sample = raw[surface_mask][["trip_id","stop_id","start_date",
                                     "arrival_time_unix","route_id"]].head(3)
        print(f"  Surface sample (first 3 rows):\n{sample.to_string()}")

    gtfs_st = load_gtfs_stop_times()

    if gtfs_st is not None and n_surface > 0:
        surface = raw[surface_mask].copy()

        # SEPTA RT trip_ids do NOT match GTFS static trip_ids — different systems.
        # Join on stop_id only, restricting to scheduled trips within ±2 hours
        # of actual arrival time-of-day to avoid matching the wrong trip.
        surface["stop_id_clean"] = (surface["stop_id"].astype(str)
                                     .str.strip()
                                     .str.replace(r"\.0$", "", regex=True))
        gtfs_st["stop_id"] = gtfs_st["stop_id"].astype(str).str.strip()

        try:
            import zoneinfo
            ET = zoneinfo.ZoneInfo("America/New_York")
        except Exception:
            import pytz
            ET = pytz.timezone("America/New_York")

        # Compute actual arrival time-of-day in seconds (Eastern Time)
        surface = surface.reset_index(drop=True)
        surface["_row_idx"] = surface.index

        collected_et = surface["collected_at"].dt.tz_convert(ET)

        def unix_to_tod_sec(unix_s):
            # GTFS scheduled times are in Eastern Time
            # Apply ET offset manually (EDT=UTC-4, EST=UTC-5)
            # Use -4 (EDT) for April–October, -5 (EST) for November–March
            # Simple approach: subtract 4h offset (EDT) — close enough for matching
            ET_OFFSET_SEC = 4 * 3600  # EDT offset (April = summer time)
            et_unix = unix_s - ET_OFFSET_SEC
            dt = pd.to_datetime(et_unix, unit="s", utc=False)
            return (dt.dt.hour * 3600 + dt.dt.minute * 60 + dt.dt.second).astype(float)

        # Use arrival_time_unix when available, else collected_at in seconds
        # NOTE: collected_at.astype(int64) gives MICROSECONDS (datetime64[us])
        # divide by 1_000_000_000 to get seconds (not 1_000_000)
        actual_unix = surface["arrival_time_unix"].fillna(
            surface["collected_at"].astype("int64") // 1_000_000_000)
        surface["actual_tod_sec"] = unix_to_tod_sec(actual_unix)

        # Merge on stop_id
        merged = surface.merge(
            gtfs_st[["stop_id","scheduled_arrival_sec"]],
            left_on="stop_id_clean",
            right_on="stop_id",
            how="left",
        )

        # Compute time difference — only keep candidates within ±2 hours (7200 sec)
        merged["tod_diff"] = (merged["scheduled_arrival_sec"]
                               - merged["actual_tod_sec"]).abs()
        merged = merged[merged["tod_diff"] <= 7200].copy()

        # Keep the closest match per original surface row
        if not merged.empty:
            best_idx = merged.groupby("_row_idx")["tod_diff"].idxmin()
            merged   = merged.loc[best_idx.dropna().astype(int)].copy()
            merged   = merged.set_index("_row_idx")
        else:
            merged = pd.DataFrame()

        # Compute delay = actual_unix − scheduled_unix
        if not merged.empty:
            # ET midnight in unix seconds.
            # pd.to_datetime(date_str, utc=False) returns datetime64[us] (microseconds)
            # → astype('int64') // 1_000_000 gives seconds (NOT // 1_000_000_000)
            ET_OFFSET_SEC     = 4 * 3600  # EDT (April–October)
            collected_et_date = collected_et.dt.strftime("%Y%m%d")
            et_midnight_unix  = (
                pd.to_datetime(collected_et_date.reindex(merged.index),
                               format="%Y%m%d", utc=False)
                .astype("int64") // 1_000_000   # microseconds → seconds
                + ET_OFFSET_SEC                  # shift midnight ET → UTC
            )
            merged["scheduled_unix"]  = et_midnight_unix + merged["scheduled_arrival_sec"]
            merged["delay_sec_gtfs"]  = actual_unix.reindex(merged.index) - merged["scheduled_unix"]
            # Sanity-clip: discard values outside ±3 hours
            merged.loc[merged["delay_sec_gtfs"].abs() > 10800, "delay_sec_gtfs"] = np.nan
            delay_series = merged["delay_sec_gtfs"].reindex(surface.index)
        else:
            delay_series = pd.Series(np.nan, index=surface.index)
        raw.loc[surface_mask, "delay_sec"] = delay_series.values
        n_surface_ok = int(delay_series.notna().sum())
        print(f"  GTFS join: {n_surface_ok:,}/{n_surface:,} bus/trolley/subway rows matched")

        if n_surface_ok == 0:
            print("  ⚠ Zero matches — check stop_id format or GTFS feed coverage")

    elif gtfs_st is None and n_surface > 0:
        print(f"  Skipping {n_surface:,} bus/trolley rows — GTFS feed unavailable")

    # Drop rows with no delay at all
    raw = raw.dropna(subset=["delay_sec"]).copy()
    raw["delay_minutes"] = (raw["delay_sec"] / 60.0).clip(DELAY_MIN_CLIP, DELAY_MAX_CLIP).round(2)

    print(f"  After delay computation: {len(raw):,} usable rows")
    print(f"    Rail (explicit delay)     : {n_rail_explicit:,}")
    print(f"    Bus/Trolley (GTFS join)   : {n_surface_ok:,}")

    # Temporal features
    raw["hour"]        = raw["collected_at"].dt.hour
    raw["day_of_week"] = raw["collected_at"].dt.dayofweek
    raw["month"]       = raw["collected_at"].dt.month
    raw["is_weekend"]  = (raw["day_of_week"] >= 5).astype(int)

    # Direction
    raw["direction_id"] = pd.to_numeric(
        raw.get("direction_id", pd.Series(dtype=float)), errors="coerce")
    raw["direction"] = raw["direction_id"].map(
        {0:"Inbound", 1:"Outbound"}).fillna("Unknown")

    # Stop sequence
    raw["stop_sequence"] = (pd.to_numeric(raw["stop_sequence"], errors="coerce")
                             .fillna(0).astype(int))

    raw["is_delayed"] = (raw["delay_minutes"] >= DELAY_THRESHOLD).astype(int)
    raw["rail_name"]  = raw["route"].apply(rail_label)
    raw["time_period"] = pd.cut(
        raw["hour"],
        bins=[0, 6, 9, 12, 16, 19, 24],
        labels=["Early Morning","AM Peak","Midday","Afternoon","PM Peak","Evening"],
        right=False,
    )

    # Deduplicate per poll window
    # Rail: stop_id is NaN (trip-level delay) → dedup on trip_id + poll_window
    # Bus/Trolley: stop_id is populated → dedup on trip_id + stop_id + poll_window
    raw["poll_window"] = raw["collected_at"].dt.floor("2min")

    rail_rows    = raw[raw["mode"] == "Regional Rail"]
    surface_rows = raw[raw["mode"] != "Regional Rail"]

    rail_deduped = (rail_rows.sort_values("collected_at")
                   .drop_duplicates(subset=["trip_id","poll_window"], keep="last"))

    surface_deduped = (surface_rows.sort_values("collected_at")
                      .drop_duplicates(subset=["trip_id","stop_id","poll_window"], keep="last"))

    raw = pd.concat([rail_deduped, surface_deduped]).sort_values("collected_at")
    raw = raw.drop(columns=["poll_window"])
    print(f"  After dedup: {len(raw):,}  "
          f"(rail={len(rail_deduped):,}  surface={len(surface_deduped):,})")

    cols = ["route","rail_name","hour","day_of_week","month","is_weekend",
            "stop_sequence","direction","delay_minutes","is_delayed","mode","time_period"]
    return raw[[c for c in cols if c in raw.columns]].reset_index(drop=True)


# ═════════════════════════════════════════════════════════════════════════
# SYNTHETIC DATASET (fallback)
# ═════════════════════════════════════════════════════════════════════════

def generate_synthetic_dataset(n: int = N_SYNTHETIC, seed: int = RANDOM_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    routes = {
        "LAN":(6.5,9.1),"CHW":(5.5,8.0),"TRE":(5.2,7.5),"FOX":(4.8,7.0),
        "WIL":(4.3,6.5),"WAR":(4.1,6.2),"MED":(3.8,5.8),"PAO":(3.5,5.5),
        "AIR":(3.2,5.2),"CHE":(2.9,4.8),"WTR":(2.5,4.5),"NOR":(2.2,4.0),
        "CYN":(1.8,3.5),
        "44":(5.8,8.2),"23":(4.5,7.0),"17":(6.2,8.8),"52":(4.0,6.5),
        "47":(3.8,6.0),"33":(3.5,5.8),"9":(4.2,6.8),"21":(3.2,5.5),
        "MFL":(1.8,3.2),"BSL":(2.1,3.8),
        "T1":(3.4,5.9),"T2":(3.8,6.0),"T3":(4.0,6.3),"T5":(3.6,5.7),
    }
    rnames = list(routes.keys())
    hp = np.array([0.02,0.04,0.08,0.09,0.07,0.05,0.05,0.05,0.05,
                   0.05,0.05,0.05,0.06,0.08,0.07,0.05,0.04,0.04,0.02])
    hp /= hp.sum()

    hour      = rng.choice(range(5,24), size=n, p=hp)
    dow       = rng.choice(range(7), size=n, p=[0.16]*5+[0.10,0.10])
    month     = rng.integers(1, 13, size=n)
    stop_seq  = rng.integers(1, 40, size=n)
    direction = rng.choice([0,1], size=n)
    ridx      = rng.choice(len(rnames), size=n)
    rcol      = [rnames[i] for i in ridx]

    mu    = np.array([routes[r][0] for r in rcol])
    sigma = np.array([routes[r][1] for r in rcol])
    peak  = np.where((hour>=7)&(hour<=9),3.5, np.where((hour>=16)&(hour<=19),4.2,0.))
    we    = np.where(dow>=5,-1.8,0.)
    win   = np.where((month==12)|(month<=2),2.1,0.)
    def_  = np.where(direction==0,0.8,0.)
    raw_d = mu + peak + we + win + def_ + stop_seq*0.08 + rng.normal(0,sigma)
    raw_d[rng.choice(n,160,replace=False)] += rng.uniform(15,45,160)
    dm = np.clip(raw_d, DELAY_MIN_CLIP, DELAY_MAX_CLIP).round(2)

    df = pd.DataFrame({
        "route":rcol, "hour":hour, "day_of_week":dow, "month":month,
        "is_weekend":(dow>=5).astype(int), "stop_sequence":stop_seq,
        "direction":np.where(direction==0,"Inbound","Outbound"),
        "delay_minutes":dm, "is_delayed":(dm>=DELAY_THRESHOLD).astype(int),
    })
    df["mode"]      = df["route"].apply(classify_mode)
    df["rail_name"] = df["route"].apply(rail_label)
    df["time_period"] = pd.cut(df["hour"], bins=[4,9,12,16,19,24],
        labels=["AM Peak","Midday","Afternoon","PM Peak","Evening"])
    return df


# ═════════════════════════════════════════════════════════════════════════
# AUTO-SELECTOR
# ═════════════════════════════════════════════════════════════════════════

def get_dataset() -> tuple:
    if os.path.exists(REAL_CSV_PATH):
        try:
            lines = sum(1 for _ in open(REAL_CSV_PATH)) - 1
            if lines > 100:
                return load_real_dataset(), "real"
            print(f"  CSV has only {lines} rows — using synthetic.")
        except Exception as e:
            print(f"  Could not load real CSV ({e}) — using synthetic.")
    else:
        print(f"  CSV not found — using synthetic.")
    return generate_synthetic_dataset(), "synthetic"


# ═════════════════════════════════════════════════════════════════════════
# STATISTICS PRINTOUT + LOG
# ═════════════════════════════════════════════════════════════════════════

def print_stats(df: pd.DataFrame, source: str) -> None:
    buf = io.StringIO()
    def out(line=""):
        buf.write(line + "\n"); print(line)

    d   = df["delay_minutes"]
    q1  = np.percentile(d,25); q3 = np.percentile(d,75); iqr = q3-q1
    z   = (d-d.mean())/d.std()
    n_out = ((d<q1-1.5*iqr)|(d>q3+1.5*iqr)).sum()

    out(f"\n{'='*62}")
    out(f"  STATISTICS  |  source: {source.upper()}  |  N={len(df):,}")
    out(f"  Generated   : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    out(f"{'='*62}")
    for label, val in [
        ("Mean",f"{d.mean():.2f} min"),("Median",f"{d.median():.2f} min"),
        ("Std dev",f"{d.std():.2f} min"),("Variance",f"{d.var(ddof=1):.2f} min²"),
        ("Min",f"{d.min():.2f} min"),("Q1",f"{q1:.2f} min"),
        ("Q3",f"{q3:.2f} min"),("Max",f"{d.max():.2f} min"),
        ("IQR",f"{iqr:.2f} min"),
        ("Outlier fence",f"> {q3+1.5*iqr:.2f} min  ({n_out} rows, {100*n_out/len(df):.1f}%)"),
        ("Skewness",f"{d.skew():.2f}"),("Kurtosis",f"{d.kurtosis():.2f}"),
        ("Z > 3",f"{(z>3).sum()} records"),
    ]:
        out(f"  {label:<20}: {val}")

    vc = df["is_delayed"].value_counts()
    out(f"\n  Class balance:")
    out(f"    On-time (0)  : {vc.get(0,0):,} ({100*vc.get(0,0)/len(df):.1f}%)")
    out(f"    Delayed (1)  : {vc.get(1,0):,} ({100*vc.get(1,0)/len(df):.1f}%)")

    def mdelay(mask): return df[mask]["delay_minutes"].mean()
    out(f"\n  Mean delay by time period:")
    out(f"    AM Peak    : {mdelay(df['hour'].isin(range(7,10))):.2f} min")
    out(f"    PM Peak    : {mdelay(df['hour'].isin(range(16,20))):.2f} min")
    out(f"    Off-peak   : {mdelay(~df['hour'].isin(list(range(7,10))+list(range(16,20)))):.2f} min")
    out(f"    Weekday    : {mdelay(df['day_of_week']<5):.2f} min")
    out(f"    Weekend    : {mdelay(df['day_of_week']>=5):.2f} min")

    out(f"\n  Mean delay by mode:")
    for m, v in df.groupby("mode")["delay_minutes"].mean().sort_values().items():
        out(f"    {m:<18}: {v:.2f} min  (n={len(df[df['mode']==m]):,})")

    for mode_label, mode_name in [("REGIONAL RAIL","Regional Rail"),
                                   ("BUS","Bus"),("SUBWAY","Subway"),("TROLLEY","Trolley")]:
        sub = df[df["mode"]==mode_name]
        if len(sub) < 5: continue
        d2 = sub["delay_minutes"]
        out(f"\n  ── {mode_label} (n={len(sub):,}) ──")
        out(f"    Mean         : {d2.mean():.2f} min")
        out(f"    Median       : {d2.median():.2f} min")
        out(f"    Std dev      : {d2.std():.2f} min")
        out(f"    Delay rate   : {100*sub['is_delayed'].mean():.1f}%")
        if len(sub["route"].unique()) > 1:
            out(f"    Worst route  : {sub.groupby('route')['delay_minutes'].mean().idxmax()}")

    out(f"\n  Pearson r with delay_minutes:")
    for col in ["hour","day_of_week","month","stop_sequence","is_delayed"]:
        if col in df.columns and df[col].nunique()>1:
            r, p = stats.pearsonr(df[col].fillna(0), d)
            sig  = "**" if p<0.001 else ("*" if p<0.05 else "n.s.")
            out(f"    {col:<18}: r={r:+.3f}  p={p:.3e}  {sig}")

    groups = [df[df["mode"]==m]["delay_minutes"].values
              for m in df["mode"].unique() if len(df[df["mode"]==m])>1]
    if len(groups)>=2:
        h, p = stats.kruskal(*groups)
        out(f"\n  Kruskal-Wallis (mode): H={h:.2f}  p={p:.3e}")

    out(f"{'='*62}\n")
    stats_path = os.path.join(OUT_DIR,"stats_report.txt")
    with open(stats_path,"w",encoding="utf-8") as f:
        f.write(buf.getvalue())
    print(f"  [OK] stats_report.txt")


# ═════════════════════════════════════════════════════════════════════════
# HELPERS
# ═════════════════════════════════════════════════════════════════════════

def _save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR,name), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  [OK] {name}")

def _hline(ax, v=DELAY_THRESHOLD):
    ax.axhline(v, color=RED, lw=1.6, linestyle="--", alpha=0.7,
               label=f"{v:.0f}-min threshold")

def _vline(ax, v=DELAY_THRESHOLD):
    ax.axvline(v, color=RED, lw=1.6, linestyle="--", alpha=0.7,
               label=f"{v:.0f}-min threshold")


# ═════════════════════════════════════════════════════════════════════════
# GLOBAL OVERVIEW FIGURES  (fig1 – fig5, fig7, fig8)
# ═════════════════════════════════════════════════════════════════════════

def fig1_histogram(df):
    fig, ax = plt.subplots(figsize=(9,5))
    ax.hist(df["delay_minutes"], bins=60, color=BLUE, edgecolor="white", alpha=0.85)
    ax.axvline(df["delay_minutes"].mean(), color=RED, lw=2.2, linestyle="--",
               label=f"Mean = {df['delay_minutes'].mean():.1f} min")
    ax.axvline(df["delay_minutes"].median(), color=GOLD, lw=2.2, linestyle=":",
               label=f"Median = {df['delay_minutes'].median():.1f} min")
    ax.axvline(DELAY_THRESHOLD, color=NAVY, lw=1.8, linestyle="-", alpha=0.7,
               label=f"{DELAY_THRESHOLD:.0f}-min threshold")
    ax.set_xlabel("Delay (minutes)", fontsize=12)
    ax.set_ylabel("Number of Stop Events", fontsize=12)
    ax.set_title("Distribution of SEPTA Transit Delays — Full Network",
                 fontsize=14, fontweight="bold", color=NAVY)
    ax.legend(fontsize=10)
    _save("fig1_delay_hist.png")

def fig2_hourly_line(df):
    h = df.groupby("hour")["delay_minutes"].agg(["mean","std","count"]).reset_index()
    h["se"] = h["std"]/np.sqrt(h["count"])
    fig, ax = plt.subplots(figsize=(10,5))
    ax.fill_between(h["hour"], h["mean"]-h["se"], h["mean"]+h["se"], alpha=0.2, color=BLUE)
    ax.plot(h["hour"], h["mean"], color=BLUE, lw=2.5, marker="o", markersize=5)
    _hline(ax)
    hrs = sorted(h["hour"].unique())
    ax.set_xticks(hrs)
    ax.set_xticklabels([f"{x}:00" for x in hrs], rotation=45, ha="right", fontsize=9)
    ax.set_xlabel("Hour of Day", fontsize=12); ax.set_ylabel("Mean Delay (minutes)", fontsize=12)
    ax.set_title("Mean Delay by Hour of Day — Full Network\n(shaded = ±1 SE)",
                 fontsize=14, fontweight="bold", color=NAVY)
    ax.legend(fontsize=10); _save("fig2_hourly_delay.png")

def fig3_mode_boxplot(df):
    order = [m for m in ["Regional Rail","Bus","Trolley","Subway"] if m in df["mode"].unique()]
    groups = [df[df["mode"]==m]["delay_minutes"].values for m in order]
    fig, ax = plt.subplots(figsize=(9,5))
    bp = ax.boxplot(groups, patch_artist=True, notch=True,
                    medianprops=dict(color="white",lw=2.5),
                    flierprops=dict(marker=".",alpha=0.3,markersize=3))
    for patch, m in zip(bp["boxes"], order):
        patch.set_facecolor(MODE_COLORS.get(m,BLUE)); patch.set_alpha(0.85)
    ax.set_xticklabels(order, fontsize=11)
    ax.set_ylabel("Delay (minutes)", fontsize=12)
    ax.set_title("Delay Distribution by Transit Mode", fontsize=14, fontweight="bold", color=NAVY)
    _hline(ax); ax.legend(fontsize=10); _save("fig3_mode_boxplot.png")

def fig4_heatmap_dow_hour(df):
    dlabels = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    pivot = df.pivot_table(values="is_delayed",index="day_of_week",
                           columns="hour",aggfunc="mean")*100
    pivot.index = [dlabels.get(i,str(i)) for i in pivot.index]
    fig, ax = plt.subplots(figsize=(13,4))
    sns.heatmap(pivot, ax=ax, cmap="YlOrRd", annot=False, linewidths=0.3,
                linecolor="white", cbar_kws={"label":"Delay Rate (%)","shrink":0.8})
    ax.set_xticklabels([f"{c}:00" for c in pivot.columns],
                       rotation=45, ha="right", fontsize=8)
    ax.set_xlabel("Hour of Day",fontsize=11); ax.set_ylabel("Day of Week",fontsize=11)
    ax.set_title("Delay Rate (%) by Hour × Day of Week — Full Network",
                 fontsize=14, fontweight="bold", color=NAVY)
    _save("fig4_heatmap.png")

def fig5_correlation_matrix(df):
    cols = [c for c in ["delay_minutes","hour","day_of_week","month",
                         "stop_sequence","is_delayed","is_weekend"] if c in df.columns]
    corr = df[cols].corr()
    mask = np.triu(np.ones_like(corr,dtype=bool))
    fig, ax = plt.subplots(figsize=(7,6))
    sns.heatmap(corr, ax=ax, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                linecolor="white", cbar_kws={"shrink":0.8})
    ax.set_title("Feature Correlation Matrix — Full Network",
                 fontsize=14, fontweight="bold", color=NAVY)
    _save("fig5_corr_heatmap.png")

def fig7_qq_plot(df):
    (osm,osr),(slope,intercept,r) = stats.probplot(df["delay_minutes"],dist="norm")
    fig, ax = plt.subplots(figsize=(6,5))
    ax.scatter(osm,osr,color=BLUE,alpha=0.3,s=4,label="Observed data")
    xl = np.array([osm.min(),osm.max()])
    ax.plot(xl,slope*xl+intercept,color=RED,lw=2,label=f"Normal ref (r={r:.3f})")
    ax.set_xlabel("Theoretical Quantiles (Normal)",fontsize=11)
    ax.set_ylabel("Sample Quantiles (delay_minutes)",fontsize=11)
    ax.set_title("Q-Q Plot: delay_minutes vs. Normal Distribution",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.legend(fontsize=10); _save("fig7_qq.png")

def fig8_scatter(df, sample_n=2_000):
    rng = np.random.default_rng(RANDOM_SEED)
    idx = rng.choice(len(df),size=min(sample_n,len(df)),replace=False)
    x = df["stop_sequence"].values[idx].astype(float)
    y = df["delay_minutes"].values[idx]
    fig, ax = plt.subplots(figsize=(7,5))
    ax.scatter(x,y,alpha=0.25,s=15,color=BLUE)
    if x.std()>0:
        m,b = np.polyfit(x,y,1)
        xs  = np.linspace(x.min(),x.max(),50)
        r,_ = stats.pearsonr(x,y)
        ax.plot(xs,m*xs+b,color=RED,lw=2.2,label=f"Trend (r={r:.3f})")
    _hline(ax)
    ax.set_xlabel("Stop Sequence Number",fontsize=11)
    ax.set_ylabel("Delay (minutes)",fontsize=11)
    ax.set_title("Scatter Plot: Stop Sequence vs. Delay",
                 fontsize=13, fontweight="bold", color=NAVY)
    ax.legend(fontsize=10); _save("fig8_scatter.png")


# ═════════════════════════════════════════════════════════════════════════
# REGIONAL RAIL FIGURES  (fig_rail_1 – fig_rail_5)
# ═════════════════════════════════════════════════════════════════════════

def _rail(df): return df[df["mode"]=="Regional Rail"].copy()

def fig_rail_1(df):
    rail = _rail(df)
    if rail.empty: return
    grp = rail.groupby("rail_name")["delay_minutes"].agg(["mean","sem","count"]).reset_index()
    grp["ci"] = 1.96*grp["sem"]
    grp = grp.sort_values("mean",ascending=True)
    colors = [RED if v>=DELAY_THRESHOLD else NAVY for v in grp["mean"]]
    fig, ax = plt.subplots(figsize=(10, max(5,len(grp)*0.55)))
    bars = ax.barh(grp["rail_name"],grp["mean"],xerr=grp["ci"],color=colors,
                   edgecolor="white",height=0.6,capsize=4,
                   error_kw={"ecolor":GRAY,"lw":1.5})
    _vline(ax)
    for bar,(_, row) in zip(bars,grp.iterrows()):
        ax.text(row["mean"]+row["ci"]+0.1, bar.get_y()+bar.get_height()/2,
                f"{row['mean']:.1f} min  (n={int(row['count']):,})", va="center", fontsize=8.5)
    ax.legend(handles=[mpatches.Patch(color=RED,label="Above 5-min threshold"),
                       mpatches.Patch(color=NAVY,label="Below 5-min threshold")],fontsize=9)
    ax.set_xlabel("Mean Delay (minutes)",fontsize=12)
    ax.set_title("Regional Rail — Mean Delay by Line (95% CI)",
                 fontsize=13, fontweight="bold", color=NAVY)
    _save("fig_rail_1.png")

def fig_rail_2(df):
    rail = _rail(df)
    if rail.empty: return
    pivot = rail.pivot_table(values="is_delayed",index="rail_name",
                              columns="hour",aggfunc="mean")*100
    fig, ax = plt.subplots(figsize=(14, max(4,len(pivot)*0.55)))
    sns.heatmap(pivot,ax=ax,cmap="YlOrRd",annot=True,fmt=".0f",
                linewidths=0.4,linecolor="white",
                cbar_kws={"label":"Delay Rate (%)","shrink":0.7})
    ax.set_xticklabels([f"{c}:00" for c in pivot.columns],
                       rotation=45,ha="right",fontsize=8)
    ax.set_xlabel("Hour of Day",fontsize=11); ax.set_ylabel("Rail Line",fontsize=11)
    ax.set_title("Regional Rail — Delay Rate (%) by Hour × Line",
                 fontsize=13,fontweight="bold",color=NAVY)
    _save("fig_rail_2.png")

def fig_rail_3(df):
    rail = _rail(df)
    if rail.empty or "direction" not in rail.columns: return
    grp = (rail[rail["direction"].isin(["Inbound","Outbound"])]
           .groupby(["rail_name","direction"])["delay_minutes"]
           .mean().unstack("direction"))
    if grp.empty: return
    grp = grp.sort_values("Inbound",ascending=False) if "Inbound" in grp else grp
    x = np.arange(len(grp)); w = 0.35
    fig, ax = plt.subplots(figsize=(12,max(5,len(grp)*0.6)))
    if "Inbound" in grp:
        ax.bar(x-w/2,grp["Inbound"],w,label="Inbound (→ Center City)",
               color=NAVY,alpha=0.85,edgecolor="white")
    if "Outbound" in grp:
        ax.bar(x+w/2,grp["Outbound"],w,label="Outbound (← Center City)",
               color=LBLUE,alpha=0.85,edgecolor="white")
    ax.set_xticks(x); ax.set_xticklabels(grp.index,rotation=30,ha="right",fontsize=9)
    _hline(ax); ax.set_ylabel("Mean Delay (minutes)",fontsize=12)
    ax.set_title("Regional Rail — Inbound vs. Outbound Mean Delay by Line",
                 fontsize=13,fontweight="bold",color=NAVY)
    ax.legend(fontsize=10); _save("fig_rail_3.png")

def fig_rail_4(df):
    rail = _rail(df)
    if rail.empty: return
    order = rail.groupby("rail_name")["delay_minutes"].mean().sort_values(ascending=False).index.tolist()
    groups = [rail[rail["rail_name"]==ln]["delay_minutes"].values for ln in order]
    fig, ax = plt.subplots(figsize=(13,5))
    bp = ax.boxplot(groups,patch_artist=True,notch=False,
                    medianprops=dict(color="white",lw=2.2),
                    flierprops=dict(marker=".",alpha=0.3,markersize=3),widths=0.55)
    cmap = plt.cm.Blues_r
    for i,patch in enumerate(bp["boxes"]):
        patch.set_facecolor(cmap(i/max(len(order),1)*0.6+0.1)); patch.set_alpha(0.9)
    ax.set_xticklabels(order,rotation=30,ha="right",fontsize=9)
    _hline(ax); ax.set_ylabel("Delay (minutes)",fontsize=12)
    ax.set_title("Regional Rail — Delay Distribution by Line (sorted by mean)",
                 fontsize=13,fontweight="bold",color=NAVY)
    ax.legend(fontsize=10); _save("fig_rail_4.png")

def fig_rail_5(df):
    rail = _rail(df)
    if rail.empty: return
    lines = (rail.groupby("rail_name")["delay_minutes"]
              .mean().sort_values(ascending=False).index.tolist()[:8])
    fig, ax = plt.subplots(figsize=(12,6))
    cmap = plt.cm.tab10
    for i,ln in enumerate(lines):
        sub = rail[rail["rail_name"]==ln].groupby("hour")["delay_minutes"].mean()
        ax.plot(sub.index,sub.values,marker="o",markersize=4,
                lw=1.8,label=ln,color=cmap(i/10))
    _hline(ax)
    hrs = sorted(rail["hour"].unique())
    ax.set_xticks(hrs)
    ax.set_xticklabels([f"{h}:00" for h in hrs],rotation=45,ha="right",fontsize=8)
    ax.set_xlabel("Hour of Day",fontsize=12); ax.set_ylabel("Mean Delay (minutes)",fontsize=12)
    ax.set_title("Regional Rail — Mean Delay by Hour per Line (top 8)",
                 fontsize=13,fontweight="bold",color=NAVY)
    ax.legend(fontsize=8,ncol=2,loc="upper right"); _save("fig_rail_5.png")


# ═════════════════════════════════════════════════════════════════════════
# BUS FIGURES  (fig_bus_1 – fig_bus_5)
# ═════════════════════════════════════════════════════════════════════════

def _bus(df): return df[df["mode"]=="Bus"].copy()

def fig_bus_1(df):
    bus = _bus(df)
    if bus.empty: print("  [SKIP] fig_bus_1 — no bus data"); return
    top = bus["route"].value_counts().nlargest(20).index
    grp = (bus[bus["route"].isin(top)].groupby("route")["delay_minutes"]
            .agg(["mean","count"]).reset_index().sort_values("mean",ascending=True))
    colors = [RED if v>=DELAY_THRESHOLD else BLUE for v in grp["mean"]]
    fig, ax = plt.subplots(figsize=(10,max(5,len(grp)*0.5)))
    bars = ax.barh(grp["route"],grp["mean"],color=colors,edgecolor="white",height=0.6)
    _vline(ax)
    for bar,(_, row) in zip(bars,grp.iterrows()):
        ax.text(row["mean"]+0.1,bar.get_y()+bar.get_height()/2,
                f"{row['mean']:.1f} min  (n={int(row['count']):,})",va="center",fontsize=8.5)
    ax.legend(handles=[mpatches.Patch(color=RED,label="Above 5-min threshold"),
                       mpatches.Patch(color=BLUE,label="Below 5-min threshold")],fontsize=9)
    ax.set_xlabel("Mean Delay (minutes)",fontsize=12)
    ax.set_title("Bus — Mean Delay by Route (top 20 by volume)",
                 fontsize=13,fontweight="bold",color=NAVY)
    _save("fig_bus_1.png")

def fig_bus_2(df):
    bus = _bus(df)
    if bus.empty: print("  [SKIP] fig_bus_2 — no bus data"); return
    wd = bus[bus["is_weekend"]==0].groupby("hour")["delay_minutes"].mean()
    we = bus[bus["is_weekend"]==1].groupby("hour")["delay_minutes"].mean()
    fig, ax = plt.subplots(figsize=(11,5))
    if not wd.empty: ax.plot(wd.index,wd.values,color=BLUE,lw=2.5,marker="o",markersize=5,label="Weekday")
    if not we.empty: ax.plot(we.index,we.values,color=GOLD,lw=2.5,marker="s",markersize=5,linestyle="--",label="Weekend")
    _hline(ax)
    hrs = sorted(bus["hour"].unique())
    ax.set_xticks(hrs); ax.set_xticklabels([f"{h}:00" for h in hrs],rotation=45,ha="right",fontsize=9)
    ax.set_xlabel("Hour of Day",fontsize=12); ax.set_ylabel("Mean Delay (minutes)",fontsize=12)
    ax.set_title("Bus — Mean Delay by Hour: Weekday vs. Weekend",
                 fontsize=13,fontweight="bold",color=NAVY)
    ax.legend(fontsize=11); _save("fig_bus_2.png")

def fig_bus_3(df):
    bus = _bus(df)
    if bus.empty: print("  [SKIP] fig_bus_3 — no bus data"); return
    top_routes = (bus.groupby("route")["delay_minutes"].mean()
                  .sort_values(ascending=False).head(10).index.tolist())
    groups = [bus[bus["route"]==r]["delay_minutes"].values for r in top_routes]
    fig, ax = plt.subplots(figsize=(12,5))
    bp = ax.boxplot(groups,patch_artist=True,notch=False,
                    medianprops=dict(color="white",lw=2.2),
                    flierprops=dict(marker=".",alpha=0.3,markersize=3),widths=0.55)
    for patch in bp["boxes"]: patch.set_facecolor(BLUE); patch.set_alpha(0.8)
    ax.set_xticklabels([f"Route {r}" for r in top_routes],rotation=30,ha="right",fontsize=9)
    _hline(ax); ax.set_ylabel("Delay (minutes)",fontsize=12)
    ax.set_title("Bus — Delay Distribution: Top 10 Most-Delayed Routes",
                 fontsize=13,fontweight="bold",color=NAVY)
    ax.legend(fontsize=10); _save("fig_bus_3.png")

def fig_bus_4(df):
    bus = _bus(df)
    if bus.empty: print("  [SKIP] fig_bus_4 — no bus data"); return
    dlabels = {0:"Mon",1:"Tue",2:"Wed",3:"Thu",4:"Fri",5:"Sat",6:"Sun"}
    pivot = bus.pivot_table(values="is_delayed",index="day_of_week",
                             columns="hour",aggfunc="mean")*100
    pivot.index = [dlabels.get(i,str(i)) for i in pivot.index]
    fig, ax = plt.subplots(figsize=(13,4))
    sns.heatmap(pivot,ax=ax,cmap="YlOrRd",annot=False,linewidths=0.3,
                linecolor="white",cbar_kws={"label":"Delay Rate (%)","shrink":0.8})
    ax.set_xticklabels([f"{c}:00" for c in pivot.columns],rotation=45,ha="right",fontsize=8)
    ax.set_xlabel("Hour of Day",fontsize=11); ax.set_ylabel("Day of Week",fontsize=11)
    ax.set_title("Bus — Delay Rate (%) by Hour × Day of Week",
                 fontsize=13,fontweight="bold",color=NAVY)
    _save("fig_bus_4.png")

def fig_bus_5(df):
    bus = _bus(df)
    if bus.empty: print("  [SKIP] fig_bus_5 — no bus data"); return
    top = bus["route"].value_counts().nlargest(20).index
    ontime = (bus[bus["route"].isin(top)].groupby("route")["is_delayed"]
              .apply(lambda x: 100*(1-x.mean())).reset_index()
              .rename(columns={"is_delayed":"ontime_pct"})
              .sort_values("ontime_pct",ascending=True))
    colors = [GREEN if v>=50 else RED for v in ontime["ontime_pct"]]
    fig, ax = plt.subplots(figsize=(10,max(5,len(ontime)*0.5)))
    bars = ax.barh(ontime["route"],ontime["ontime_pct"],color=colors,edgecolor="white",height=0.6)
    ax.axvline(50,color=NAVY,lw=1.6,linestyle="--",alpha=0.7)
    for bar,(_, row) in zip(bars,ontime.iterrows()):
        ax.text(row["ontime_pct"]+0.5,bar.get_y()+bar.get_height()/2,
                f"{row['ontime_pct']:.1f}%",va="center",fontsize=8.5)
    ax.legend(handles=[mpatches.Patch(color=GREEN,label="≥50% on-time"),
                       mpatches.Patch(color=RED,label="<50% on-time")],fontsize=9)
    ax.set_xlabel("On-Time Rate (%)",fontsize=12)
    ax.set_title("Bus — On-Time Rate (%) by Route (top 20 by volume)",
                 fontsize=13,fontweight="bold",color=NAVY)
    _save("fig_bus_5.png")


# ═════════════════════════════════════════════════════════════════════════
# SUBWAY & TROLLEY FIGURES  (fig_subway_1 – fig_subway_5)
# ═════════════════════════════════════════════════════════════════════════

def _st(df): return df[df["mode"].isin(["Subway","Trolley"])].copy()

def fig_subway_1(df):
    st = _st(df)
    if st.empty: print("  [SKIP] fig_subway_1 — no subway/trolley data"); return
    subway_r  = sorted(st[st["mode"]=="Subway"]["route"].unique())
    trolley_r = sorted(st[st["mode"]=="Trolley"]["route"].unique())
    order  = subway_r + trolley_r
    if not order: return
    groups = [st[st["route"]==r]["delay_minutes"].values for r in order]
    colors = [MODE_COLORS["Subway"]]*len(subway_r)+[MODE_COLORS["Trolley"]]*len(trolley_r)
    fig, ax = plt.subplots(figsize=(max(8,len(order)*1.1),5))
    bp = ax.boxplot(groups,patch_artist=True,notch=False,
                    medianprops=dict(color="white",lw=2.2),
                    flierprops=dict(marker=".",alpha=0.3,markersize=3),widths=0.55)
    for patch, color in zip(bp["boxes"],colors):
        patch.set_facecolor(color); patch.set_alpha(0.85)
    ax.set_xticklabels(order,rotation=30,ha="right",fontsize=9)
    _hline(ax); ax.set_ylabel("Delay (minutes)",fontsize=12)
    ax.set_title("Subway & Trolley — Delay Distribution by Route",
                 fontsize=13,fontweight="bold",color=NAVY)
    ax.legend(handles=[mpatches.Patch(color=MODE_COLORS["Subway"],label="Subway"),
                       mpatches.Patch(color=MODE_COLORS["Trolley"],label="Trolley")],fontsize=10)
    _save("fig_subway_1.png")

def fig_subway_2(df):
    st = _st(df)
    if st.empty: print("  [SKIP] fig_subway_2 — no subway/trolley data"); return
    grp = (st.groupby(["route","mode"])["delay_minutes"]
             .agg(["mean","count"]).reset_index().sort_values("mean",ascending=True))
    colors = [MODE_COLORS.get(m,GRAY) for m in grp["mode"]]
    fig, ax = plt.subplots(figsize=(10,max(4,len(grp)*0.55)))
    bars = ax.barh(grp["route"],grp["mean"],color=colors,edgecolor="white",height=0.6)
    _vline(ax)
    for bar,(_, row) in zip(bars,grp.iterrows()):
        ax.text(row["mean"]+0.1,bar.get_y()+bar.get_height()/2,
                f"{row['mean']:.1f} min  (n={int(row['count']):,})",va="center",fontsize=8.5)
    ax.legend(handles=[mpatches.Patch(color=MODE_COLORS["Subway"],label="Subway"),
                       mpatches.Patch(color=MODE_COLORS["Trolley"],label="Trolley")],fontsize=10)
    ax.set_xlabel("Mean Delay (minutes)",fontsize=12)
    ax.set_title("Subway & Trolley — Mean Delay by Route",
                 fontsize=13,fontweight="bold",color=NAVY)
    _save("fig_subway_2.png")

def fig_subway_3(df):
    st = _st(df)
    if st.empty: print("  [SKIP] fig_subway_3 — no subway/trolley data"); return
    fig, ax = plt.subplots(figsize=(11,5))
    for mode_name, color, marker in [("Subway",GREEN,"o"),("Trolley",GOLD,"s")]:
        sub = st[st["mode"]==mode_name].groupby("hour")["delay_minutes"].mean()
        if not sub.empty:
            ax.plot(sub.index,sub.values,color=color,lw=2.5,marker=marker,markersize=5,label=mode_name)
    _hline(ax)
    hrs = sorted(st["hour"].unique())
    ax.set_xticks(hrs); ax.set_xticklabels([f"{h}:00" for h in hrs],rotation=45,ha="right",fontsize=9)
    ax.set_xlabel("Hour of Day",fontsize=12); ax.set_ylabel("Mean Delay (minutes)",fontsize=12)
    ax.set_title("Subway vs. Trolley — Mean Delay by Hour of Day",
                 fontsize=13,fontweight="bold",color=NAVY)
    ax.legend(fontsize=11); _save("fig_subway_3.png")

def fig_subway_4(df):
    st = _st(df)
    if st.empty: print("  [SKIP] fig_subway_4 — no subway/trolley data"); return
    grp = (st.groupby(["mode","is_weekend"])["delay_minutes"]
             .mean().unstack("is_weekend")
             .rename(columns={0:"Weekday",1:"Weekend"}))
    modes = grp.index.tolist()
    x = np.arange(len(modes)); w = 0.35
    fig, ax = plt.subplots(figsize=(8,5))
    ax.bar(x-w/2, grp.get("Weekday",[0]*len(modes)), w, label="Weekday",
           color=[MODE_COLORS.get(m,GRAY) for m in modes], alpha=0.9, edgecolor="white")
    if "Weekend" in grp:
        ax.bar(x+w/2, grp["Weekend"], w, label="Weekend",
               color=[MODE_COLORS.get(m,GRAY) for m in modes],
               alpha=0.45, edgecolor="white", hatch="//")
    ax.set_xticks(x); ax.set_xticklabels(modes,fontsize=11)
    _hline(ax); ax.set_ylabel("Mean Delay (minutes)",fontsize=12)
    ax.set_title("Subway & Trolley — Weekday vs. Weekend Mean Delay",
                 fontsize=13,fontweight="bold",color=NAVY)
    ax.legend(fontsize=10); _save("fig_subway_4.png")

def fig_subway_5(df):
    st = _st(df)
    if st.empty: print("  [SKIP] fig_subway_5 — no subway/trolley data"); return
    pivot = st.pivot_table(values="is_delayed",index="route",
                            columns="hour",aggfunc="mean")*100
    fig, ax = plt.subplots(figsize=(13,max(3,len(pivot)*0.55)))
    sns.heatmap(pivot,ax=ax,cmap="YlOrRd",annot=True,fmt=".0f",
                linewidths=0.4,linecolor="white",
                cbar_kws={"label":"Delay Rate (%)","shrink":0.7})
    ax.set_xticklabels([f"{c}:00" for c in pivot.columns],rotation=45,ha="right",fontsize=8)
    ax.set_xlabel("Hour of Day",fontsize=11); ax.set_ylabel("Route",fontsize=11)
    ax.set_title("Subway & Trolley — Delay Rate (%) by Hour × Route",
                 fontsize=13,fontweight="bold",color=NAVY)
    _save("fig_subway_5.png")


# ═════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════

def main():
    print("\nSEPTA Deliverable 2 — analysis.py")
    print("----------------------------------")

    print("Step 1: Loading dataset ...")
    df, source = get_dataset()
    print(f"         {len(df):,} records  |  source = {source}")
    for m in ["Regional Rail","Bus","Subway","Trolley"]:
        n = len(df[df["mode"]==m])
        if n > 0:
            print(f"           {m:<18}: {n:,} ({100*n/len(df):.1f}%)")

    if len(df) < 10:
        print("  Not enough data to generate figures. Exiting.")
        return

    print("\nStep 2: Statistics ...")
    print_stats(df, source)

    print("\nStep 3: Global overview figures ...")
    fig1_histogram(df); fig2_hourly_line(df); fig3_mode_boxplot(df)
    fig4_heatmap_dow_hour(df); fig5_correlation_matrix(df)
    fig7_qq_plot(df); fig8_scatter(df)

    print("\nStep 4: Regional Rail figures ...")
    fig_rail_1(df); fig_rail_2(df); fig_rail_3(df); fig_rail_4(df); fig_rail_5(df)

    print("\nStep 5: Bus figures ...")
    fig_bus_1(df); fig_bus_2(df); fig_bus_3(df); fig_bus_4(df); fig_bus_5(df)

    print("\nStep 6: Subway & Trolley figures ...")
    fig_subway_1(df); fig_subway_2(df); fig_subway_3(df); fig_subway_4(df); fig_subway_5(df)

    n_modes_with_data = sum(len(df[df["mode"]==m])>0
                            for m in ["Regional Rail","Bus","Subway","Trolley"])
    print(f"\n{'='*52}")
    print(f"  Done — 22 figures + stats_report.txt")
    print(f"  Source : {source.upper()}")
    print(f"  Modes  : {n_modes_with_data}/4 have data")
    print(f"  Output : {OUT_DIR}")
    print(f"{'='*52}\n")


if __name__ == "__main__":
    main()
