"""
download_gtfs.py
================
SEPTA Transit Delay Prediction — One-Time GTFS Static Feed Downloader
CISC 520 Data Mining and Engineering, Harrisburg University

PURPOSE
-------
Downloads the SEPTA GTFS static schedule (bus/trolley/subway timetable)
and extracts stop_times.txt into E:\\SEPTA_data\\gtfs\\

This is required ONCE before running analysis.py with real data.
Without it, bus and trolley delay computation cannot proceed.

RUN THIS ON YOUR PC (not inside Claude's sandbox):
    python download_gtfs.py

WHAT IT DOES
------------
1. Downloads gtfs_public.zip from SEPTA's GitHub releases (~15 MB)
2. Extracts the inner google_bus.zip
3. Extracts stop_times.txt from google_bus.zip into E:\\SEPTA_data\\gtfs\\
4. Prints a summary of the stop_times table so you can verify it worked

DEPENDENCIES
------------
    pip install requests   (standard library zipfile is used for extraction)

OUTPUT
------
    E:\\SEPTA_data\\gtfs\\stop_times.txt   (~200 MB uncompressed)
"""

import os
import io
import sys
import zipfile

try:
    import requests
except ImportError:
    print("ERROR: requests not installed. Run: pip install requests")
    sys.exit(1)

# ── Configuration ─────────────────────────────────────────────────────────

GTFS_OUT_DIR = r"E:\SEPTA_data\gtfs"
GTFS_URL     = "https://github.com/septadev/GTFS/releases/download/v202602220/gtfs_public.zip"

# ── Main ──────────────────────────────────────────────────────────────────

def main():
    os.makedirs(GTFS_OUT_DIR, exist_ok=True)
    stop_times_path = os.path.join(GTFS_OUT_DIR, "stop_times.txt")

    if os.path.exists(stop_times_path):
        size_mb = os.path.getsize(stop_times_path) / 1_048_576
        print(f"stop_times.txt already exists ({size_mb:.0f} MB).")
        print("Delete it and re-run if you need a fresh download.")
        return

    print(f"Downloading GTFS static feed from:\n  {GTFS_URL}\n")
    print("This may take 30–60 seconds on a typical connection...")

    try:
        resp = requests.get(GTFS_URL, stream=True, timeout=120)
        resp.raise_for_status()
    except requests.exceptions.RequestException as e:
        print(f"\nERROR downloading GTFS feed: {e}")
        print("\nAlternative: download manually from:")
        print("  https://github.com/septadev/GTFS/releases")
        print(f"Save gtfs_public.zip to {GTFS_OUT_DIR} and re-run this script.")
        sys.exit(1)

    # Load into memory (outer zip ~15 MB — manageable)
    total = int(resp.headers.get("content-length", 0))
    data  = b""
    for chunk in resp.iter_content(chunk_size=65536):
        data += chunk
        if total:
            pct = 100 * len(data) / total
            print(f"  Downloaded {len(data)/1_048_576:.1f}/{total/1_048_576:.1f} MB  ({pct:.0f}%)",
                  end="\r")
    print(f"\nDownloaded {len(data)/1_048_576:.1f} MB total.")

    # Open outer zip
    print("\nExtracting stop_times.txt ...")
    try:
        with zipfile.ZipFile(io.BytesIO(data)) as outer:
            names = outer.namelist()

            # Case 1: gtfs_public.zip contains google_bus.zip (nested)
            if "google_bus.zip" in names:
                print("  Found nested google_bus.zip — extracting inner zip...")
                inner_data = outer.read("google_bus.zip")
                with zipfile.ZipFile(io.BytesIO(inner_data)) as inner:
                    inner.extract("stop_times.txt", GTFS_OUT_DIR)

            # Case 2: stop_times.txt directly in the zip
            elif "stop_times.txt" in names:
                outer.extract("stop_times.txt", GTFS_OUT_DIR)

            # Case 3: stop_times.txt in a subfolder
            else:
                st_name = next((n for n in names if n.endswith("stop_times.txt")), None)
                if st_name:
                    data_bytes = outer.read(st_name)
                    with open(stop_times_path, "wb") as f:
                        f.write(data_bytes)
                else:
                    print(f"ERROR: stop_times.txt not found in zip. Contents:")
                    for n in names[:20]:
                        print(f"  {n}")
                    sys.exit(1)

    except zipfile.BadZipFile as e:
        print(f"ERROR: Downloaded file is not a valid zip: {e}")
        sys.exit(1)

    # Verify
    if not os.path.exists(stop_times_path):
        print("ERROR: Extraction seemed to succeed but stop_times.txt not found.")
        sys.exit(1)

    size_mb = os.path.getsize(stop_times_path) / 1_048_576
    print(f"\n✓ stop_times.txt saved to: {stop_times_path}")
    print(f"  File size: {size_mb:.0f} MB")

    # Quick preview
    print("\nFirst 3 rows of stop_times.txt:")
    with open(stop_times_path, encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(f"  {line.rstrip()}")
            if i >= 3:
                break

    print("\n✓ GTFS static feed ready.")
    print("  You can now run:  python analysis.py")


if __name__ == "__main__":
    main()
