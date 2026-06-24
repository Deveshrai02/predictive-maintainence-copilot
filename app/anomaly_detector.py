"""Anomaly detection over C-MAPSS turbofan sensor data.

Plain-English overview
-----------------------
This module is the "health monitor" for our equipment. We pretend each
turbofan engine in the CMAPSS dataset is a real piece of plant equipment.
For any equipment ID you give it, it answers three simple questions:

    1. How much life does this machine have left?  (the RUL estimate)
    2. Is it currently in trouble?                  (anomaly_detected)
    3. How bad is it?                               (normal / warning / critical)

It reads the cleaned sensor file ONCE when the module is first imported,
keeps it in memory, and then just looks things up — so calls are fast.
"""

from datetime import datetime
import os

import pandas as pd

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
# Where the cleaned data lives. We build an absolute path off this file's
# location so the module works no matter which folder you run it from.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(_THIS_DIR, "..", "data", "processed", "cmapss_processed.csv")

# The 7 sensors we kept during preprocessing. We list them here so the
# "sensor_summary" always reports the same set in the same order.
KEY_SENSORS = ["s2", "s3", "s4", "s7", "s11", "s12", "s15"]

# --------------------------------------------------------------------------- #
# Severity thresholds — see the rationale comment below.
# --------------------------------------------------------------------------- #
# Why two thresholds instead of one on/off alarm?
#   A single "broken / not broken" flag gives the operator no time to react.
#   Splitting the danger zone into two bands turns the alarm into a traffic
#   light, so people can plan instead of scramble:
#
#     normal   (RUL > 30) : machine is healthy, nothing to do.
#     warning  (15-30 RUL): degradation is now clearly visible in the sensors.
#                           There is still a comfortable buffer of cycles, so
#                           this is the "schedule maintenance soon, order the
#                           part" stage — act on your own timetable.
#     critical (RUL < 15) : failure is close. The buffer is almost gone, so
#                           this is the "stop and intervene now" stage —
#                           unplanned downtime is imminent if you wait.
#
#   Why 30 and 15 specifically?  30 matches the "anomaly" label we created in
#   preprocessing (the last ~15-25% of an engine's life, where the fault
#   signature actually becomes detectable). 15 splits that danger window
#   roughly in half: the first half is early-warning lead time, the second
#   half is the act-now zone. The numbers are deliberately conservative so we
#   warn early rather than risk a surprise failure.
WARNING_RUL = 30   # at or below this (and above CRITICAL) => "warning"
CRITICAL_RUL = 15  # below this => "critical"


# --------------------------------------------------------------------------- #
# Load the data once, at import time.
# --------------------------------------------------------------------------- #
# Reading the CSV is the slow part, so we do it a single time and hold the
# result in this module-level variable. Every function below reuses it.
_DATA = pd.read_csv(DATA_PATH)


def _latest_row(equipment_id: str):
    """Return the most recent reading for one machine, or None if unknown.

    In the dataset each engine is recorded cycle by cycle until it fails.
    The row with the highest 'cycle' number is therefore the newest reading
    — the machine's "right now" snapshot.
    """
    # Keep only the rows for this machine. (engine_id is our stand-in for a
    # real equipment ID, so we compare against it as a string to be forgiving
    # about whether the caller passes "5" or 5.)
    subset = _DATA[_DATA["engine_id"].astype(str) == str(equipment_id)]
    if subset.empty:
        return None
    # The newest reading is the one with the largest cycle count.
    return subset.loc[subset["cycle"].idxmax()]


def _severity_from_rul(rul: float) -> str:
    """Turn a remaining-life number into a traffic-light label."""
    if rul < CRITICAL_RUL:
        return "critical"
    if rul <= WARNING_RUL:
        return "warning"
    return "normal"


def check_anomaly(equipment_id: str) -> dict:
    """Give a one-shot health report for a single machine.

    Returns a dictionary the agent (or a dashboard) can read directly.
    """
    row = _latest_row(equipment_id)

    # Case 1: we have never heard of this machine. Be explicit rather than
    # crashing, so the caller can handle it gracefully.
    if row is None:
        return {
            "equipment_id": equipment_id,
            "anomaly_detected": False,
            "note": "Unknown equipment ID — no data on record for this machine.",
        }

    # Pull the values we care about out of the latest reading.
    rul = int(row["RUL"])                      # cycles of life left
    is_anomaly = bool(row["anomaly"])          # the 1/0 flag from preprocessing
    severity = _severity_from_rul(rul)

    # A small dictionary of the 7 sensor values, e.g. {"s2": 642.1, ...}.
    # This lets the agent see the actual readings, not just the verdict.
    sensor_summary = {s: float(row[s]) for s in KEY_SENSORS}

    return {
        "equipment_id": equipment_id,
        "current_rul_estimate": rul,
        "anomaly_detected": is_anomaly,
        "anomaly_severity": severity,
        "sensor_summary": sensor_summary,
        # The dataset has no real clock, so we stamp the moment of the check.
        # In a live system this would be the time the reading arrived.
        "last_updated": datetime.now().isoformat(timespec="seconds"),
    }


def get_recent_trend(equipment_id: str, cycles: int = 10) -> dict:
    """Show the last N readings so we can tell 'getting worse' from 'steady'.

    A single snapshot tells you the current state; a trend tells you the
    *direction*. If sensor values are drifting cycle over cycle, the machine
    is degrading; if they are flat, it is stable. The agent uses this to
    reason about urgency.
    """
    subset = _DATA[_DATA["engine_id"].astype(str) == str(equipment_id)]
    if subset.empty:
        return {
            "equipment_id": equipment_id,
            "note": "Unknown equipment ID — no data on record for this machine.",
            "trend": [],
        }

    # Sort oldest -> newest by cycle, then take the final `cycles` rows.
    # tail(n) returns the last n rows, which are the most recent readings.
    recent = subset.sort_values("cycle").tail(cycles)

    # Turn each row into a small dict of {cycle, RUL, and the 7 sensors}.
    # The result is a list ordered oldest-first, so reading top-to-bottom
    # shows how things changed over time.
    columns = ["cycle", "RUL"] + KEY_SENSORS
    trend = recent[columns].to_dict(orient="records")

    return {
        "equipment_id": equipment_id,
        "cycles_returned": len(trend),
        "trend": trend,
    }


# --------------------------------------------------------------------------- #
# Quick manual smoke test: `python app/anomaly_detector.py`
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import json

    # Engine 1 exists; "999" should not.
    print("check_anomaly('1'):")
    print(json.dumps(check_anomaly("1"), indent=2))
    print("\ncheck_anomaly('999'):")
    print(json.dumps(check_anomaly("999"), indent=2))
    print("\nget_recent_trend('1', cycles=5):")
    print(json.dumps(get_recent_trend("1", cycles=5), indent=2))
