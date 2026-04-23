r"""
collect_septa.py
================
SEPTA Transit Delay Prediction — Real-Time Data Collector
CISC 520 Data Mining and Engineering, Harrisburg University

PURPOSE
-------
Polls the two SEPTA GTFS-RT TripUpdate endpoints (bus/trolley and
regional rail) every 90 seconds and appends one CSV row per
stop-time-update event to E:\\SEPTA_data\septa_realtime_raw.csv.

HOW TO RUN
----------
1. Create the output folder (one-time):
       mkdir E:\\SEPTA_data

2. Install dependencies (one-time):
       pip install requests gtfs-realtime-bindings protobuf

3. Copy this file to E:\\SEPTA_data\ then start collecting:
       cd E:\\SEPTA_data
       python collect_septa.py

   The script appends to septa_realtime_raw.csv in E:\\SEPTA_data\.
   Safe to Ctrl+C and restart — it resumes without duplicating data.

EXPECTED FILE SIZES
-------------------
   2  hours  →  ~15  MB   (~80,000  rows)
   12 hours  →  ~90  MB   (~480,000 rows)
   48 hours  →  ~360 MB   (~1.9M    rows)   

OUTPUTS
-------
septa_realtime_raw.csv   — one row per stop-time-update poll result
collect_septa.log        — full activity log with timestamps

OUTPUT CSV COLUMNS
------------------
collected_at        ISO-8601 timestamp of the API poll
feed_type           'bus_trolley' or 'rail'
trip_id             GTFS trip identifier
route_id            Route (e.g. '44', 'MFL', 'LAN')
direction_id        0 or 1
start_date          Service date (YYYYMMDD)
stop_id             GTFS stop identifier
stop_sequence       Position of this stop along the trip
arrival_delay_sec   Reported arrival delay in seconds (positive = late)
departure_delay_sec Reported departure delay in seconds
arrival_time_unix   Predicted arrival time (Unix timestamp), if given
departure_time_unix Predicted departure time (Unix timestamp), if given
schedule_rel        SCHEDULED / SKIPPED / NO_DATA
vehicle_id          Vehicle identifier (if provided)

DELAY CALCULATION
-----------------
delay_minutes = arrival_delay_sec / 60
When arrival_delay_sec is absent, departure_delay_sec is used as fallback.
Positive = late, Negative = early, 0 = on time.

NOTE ON SEPTA ENDPOINTS
-----------------------
SEPTA publishes two separate GTFS-RT TripUpdate feeds:
  Bus + Trolley:  https://www3.septa.org/gtfsrt/TripUpdate.pb
  Regional Rail:  https://www3.septa.org/gtfsrt/rail/TripUpdate.pb

Both return binary protobuf (protocol buffer) data.
The gtfs-realtime-bindings package parses this into Python objects.
"""

# ── Standard library ──────────────────────────────────────────────────────
import csv
import logging
import os
import sys
import time
from datetime import datetime, timezone

# ── Third-party ───────────────────────────────────────────────────────────
try:
    import requests
    from google.transit import gtfs_realtime_pb2
except ImportError:
    print(
        "\n[ERROR] Missing dependencies. Run:\n"
        "    pip install requests gtfs-realtime-bindings protobuf\n"
    )
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────

# SEPTA GTFS-RT TripUpdate feed URLs  (verified April 2026 from data.gov catalog)
FEEDS = {
    "bus_trolley": "https://www3.septa.org/gtfsrt/septa-pa-us/Trip/rtTripUpdates.pb",
    "rail":        "https://www3.septa.org/gtfsrt/septarail-pa-us/Trip/rtTripUpdates.pb",
}

POLL_INTERVAL_SEC = 90      # seconds between each full poll cycle
REQUEST_TIMEOUT   = 15      # seconds before giving up on one HTTP request
MAX_RETRIES       = 3       # retry attempts per feed per poll
RETRY_WAIT_SEC    = 10      # wait between retries

# ── Output location ───────────────────────────────────────────────────────
# All output goes to E:\\SEPTA_data\ on your portal drive.
# The folder is created automatically if it does not exist.
OUT_DIR  = r"E:\\SEPTA_data"
os.makedirs(OUT_DIR, exist_ok=True)   # creates the folder if missing

CSV_PATH = os.path.join(OUT_DIR, "septa_realtime_raw.csv")
LOG_PATH = os.path.join(OUT_DIR, "collect_septa.log")

CSV_COLUMNS = [
    "collected_at",
    "feed_type",
    "trip_id",
    "route_id",
    "direction_id",
    "start_date",
    "stop_id",
    "stop_sequence",
    "arrival_delay_sec",
    "departure_delay_sec",
    "arrival_time_unix",
    "departure_time_unix",
    "schedule_rel",
    "vehicle_id",
]


# ─────────────────────────────────────────────────────────────────────────
# Logging setup
# ─────────────────────────────────────────────────────────────────────────

def setup_logging():
    fmt = "%(asctime)s  %(levelname)-8s  %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(LOG_PATH, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────

def open_csv(path: str):
    """
    Open the CSV for appending.  Write the header row only if the file
    is new or empty (so resuming after a restart doesn't duplicate the header).
    Returns the (file_handle, csv.DictWriter) tuple.
    """
    file_is_new = not os.path.exists(path) or os.path.getsize(path) == 0
    fh = open(path, "a", newline="", encoding="utf-8")
    writer = csv.DictWriter(fh, fieldnames=CSV_COLUMNS)
    if file_is_new:
        writer.writeheader()
        log.info("Created new CSV: %s", path)
    else:
        log.info("Appending to existing CSV: %s", path)
    return fh, writer


# ─────────────────────────────────────────────────────────────────────────
# Feed fetching
# ─────────────────────────────────────────────────────────────────────────

def fetch_feed(url: str, feed_type: str) -> gtfs_realtime_pb2.FeedMessage | None:
    """
    Fetch and parse one GTFS-RT protobuf feed.
    Returns a parsed FeedMessage, or None on permanent failure.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, timeout=REQUEST_TIMEOUT)
            resp.raise_for_status()
            msg = gtfs_realtime_pb2.FeedMessage()
            msg.ParseFromString(resp.content)
            return msg
        except requests.exceptions.HTTPError as e:
            log.warning("[%s] HTTP error (attempt %d/%d): %s",
                        feed_type, attempt, MAX_RETRIES, e)
        except requests.exceptions.ConnectionError as e:
            log.warning("[%s] Connection error (attempt %d/%d): %s",
                        feed_type, attempt, MAX_RETRIES, e)
        except requests.exceptions.Timeout:
            log.warning("[%s] Timeout (attempt %d/%d)", feed_type, attempt, MAX_RETRIES)
        except Exception as e:
            log.error("[%s] Unexpected error: %s", feed_type, e)
            return None   # don't retry unexpected errors

        if attempt < MAX_RETRIES:
            time.sleep(RETRY_WAIT_SEC)

    log.error("[%s] All %d retries failed — skipping this poll cycle.",
              feed_type, MAX_RETRIES)
    return None


# ─────────────────────────────────────────────────────────────────────────
# Filtering strategy
# ─────────────────────────────────────────────────────────────────────────
#
# SEPTA's bus feed returns ALL remaining stops for every active trip —
# ~350 stops × 65 active routes ≈ 23,000 rows per poll.  Only the FIRST
# stop-time-update in each trip is an observed delay (the vehicle is
# actively approaching that stop).  Later stops are schedule projections.
#
# Setting NEXT_STOP_ONLY = True keeps only the first STU per trip,
# reducing rows to ~400–700 per poll — manageable over 48 hours (~40 MB).
# Set to False to capture full trip projections (warning: ~8 GB over 48h).
NEXT_STOP_ONLY = True


# ─────────────────────────────────────────────────────────────────────────
# Feed parsing
# ─────────────────────────────────────────────────────────────────────────

# Map the protobuf ScheduleRelationship int to a readable string
SCHEDULE_REL_MAP = {
    0: "SCHEDULED",
    1: "SKIPPED",
    2: "NO_DATA",
    3: "UNSCHEDULED",
    5: "DUPLICATED",
}


def parse_feed(
    msg: gtfs_realtime_pb2.FeedMessage,
    feed_type: str,
    collected_at: str,
) -> list[dict]:
    """
    Extract one row per StopTimeUpdate from a parsed FeedMessage.
    Captures rows whether delay is reported as:
      - an explicit 'delay' field (seconds relative to schedule), or
      - an absolute 'time' field (unix timestamp — we store it and leave
        delay blank; analysis.py will compute it against the GTFS schedule)
    """
    rows = []

    for entity in msg.entity:
        if not entity.HasField("trip_update"):
            continue

        tu    = entity.trip_update
        trip  = tu.trip
        veh   = tu.vehicle if tu.HasField("vehicle") else None

        trip_id      = trip.trip_id      or ""
        route_id     = trip.route_id     or ""
        direction_id = trip.direction_id if trip.HasField("direction_id") else ""
        vehicle_id   = veh.id if veh else ""

        # start_date: use protobuf field if present, else derive from
        # current Eastern Time date (SEPTA schedules run on ET service days)
        if trip.start_date:
            start_date = trip.start_date
        else:
            # collected_at is already the poll timestamp — derive ET date from it
            from datetime import datetime, timezone, timedelta
            # ET = UTC-5 (EST) or UTC-4 (EDT); use simple offset for robustness
            # (avoids requiring zoneinfo/pytz on the collection machine)
            utc_now = datetime.now(timezone.utc)
            # EST offset; DST adds 1hr — approximation is fine for date boundary
            et_now  = utc_now - timedelta(hours=4)   # EDT (Apr–Oct)
            start_date = et_now.strftime("%Y%m%d")

        stop_updates = tu.stop_time_update
        # When NEXT_STOP_ONLY is True, only process the first STU per trip.
        # This is the stop the vehicle is actively approaching — the only
        # one with a real observed delay rather than a schedule projection.
        if NEXT_STOP_ONLY and len(stop_updates) > 0:
            stop_updates = stop_updates[:1]

        for stu in stop_updates:
            arr_delay = ""
            arr_time  = ""
            if stu.HasField("arrival"):
                if stu.arrival.HasField("delay"):
                    arr_delay = stu.arrival.delay
                if stu.arrival.HasField("time"):
                    arr_time = stu.arrival.time

            dep_delay = ""
            dep_time  = ""
            if stu.HasField("departure"):
                if stu.departure.HasField("delay"):
                    dep_delay = stu.departure.delay
                if stu.departure.HasField("time"):
                    dep_time = stu.departure.time

            # Keep the row if we have ANY timing information
            # (delay OR absolute time — don't discard time-only rows)
            has_data = any(v != "" for v in [arr_delay, dep_delay, arr_time, dep_time])
            if not has_data:
                continue

            sched_rel = SCHEDULE_REL_MAP.get(
                stu.schedule_relationship, str(stu.schedule_relationship)
            )

            rows.append({
                "collected_at":        collected_at,
                "feed_type":           feed_type,
                "trip_id":             trip_id,
                "route_id":            route_id,
                "direction_id":        direction_id,
                "start_date":          start_date,
                "stop_id":             stu.stop_id,
                "stop_sequence":       stu.stop_sequence if stu.HasField("stop_sequence") else "",
                "arrival_delay_sec":   arr_delay,
                "departure_delay_sec": dep_delay,
                "arrival_time_unix":   arr_time,
                "departure_time_unix": dep_time,
                "schedule_rel":        sched_rel,
                "vehicle_id":          vehicle_id,
            })

    return rows


# ─────────────────────────────────────────────────────────────────────────
# Main collection loop
# ─────────────────────────────────────────────────────────────────────────

def main():
    setup_logging()
    log.info("=" * 60)
    log.info("SEPTA Real-Time Collector starting")
    log.info("CSV output : %s", CSV_PATH)
    log.info("Poll interval: %d seconds", POLL_INTERVAL_SEC)
    log.info("Feeds: %s", list(FEEDS.keys()))
    log.info("=" * 60)

    fh, writer = open_csv(CSV_PATH)
    total_rows  = 0
    poll_count  = 0

    try:
        while True:
            poll_start   = time.monotonic()
            collected_at = datetime.now(timezone.utc).isoformat(timespec="seconds")
            poll_count  += 1
            batch_rows   = 0
            feed_counts  = {k: 0 for k in FEEDS}

            for feed_type, url in FEEDS.items():
                msg = fetch_feed(url, feed_type)
                if msg is None:
                    continue

                rows = parse_feed(msg, feed_type, collected_at)
                for row in rows:
                    writer.writerow(row)
                batch_rows += len(rows)
                feed_counts[feed_type] = len(rows)

            fh.flush()
            total_rows += batch_rows

            elapsed = time.monotonic() - poll_start
            counts_str = "  ".join(f"{k}={v}" for k, v in feed_counts.items())
            log.info(
                "Poll #%d  |  +%d rows (%s)  |  %d total  |  %.1f s",
                poll_count, batch_rows, counts_str, total_rows, elapsed,
            )

            # Sleep for the remainder of the interval
            sleep_for = max(0, POLL_INTERVAL_SEC - elapsed)
            time.sleep(sleep_for)

    except KeyboardInterrupt:
        log.info("Interrupted by user.  %d total rows written.", total_rows)
    finally:
        fh.close()
        log.info("CSV closed: %s", CSV_PATH)


if __name__ == "__main__":
    main()
