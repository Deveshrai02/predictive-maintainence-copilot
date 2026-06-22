"""
Preprocess the NASA CMAPSS FD001 training set into a clean, labeled CSV
for an anomaly / predictive-maintenance model.

Pipeline (each step is also commented inline below):
  1. Load the raw space-separated turbofan run-to-failure data.
  2. Compute Remaining Useful Life (RUL) per row.
  3. Add a binary "anomaly" label for the degrading (last-30-cycles) regime.
  4. Keep only the 7 most informative, low-noise sensors.
  5. Export a tidy CSV.
  6. Print a quick summary.

Background for interview context:
  CMAPSS FD001 is a "run-to-failure" dataset. Each engine_id is one engine
  that is monitored cycle-by-cycle until it fails. The last recorded cycle
  for an engine IS the failure point. The raw file has no RUL column — we
  derive it ourselves from the cycle counts, which is the standard approach.
"""

import os
import pandas as pd

# --- File locations -------------------------------------------------------
RAW_PATH = "data/cmapss/train_FD001.txt"
OUT_PATH = "data/processed/cmapss_processed.csv"

# The 7 sensors we keep. In FD001 these are the ones that show a clear,
# monotonic trend as the engine degrades and have low measurement noise.
# The other sensors are either flat (constant) or dominated by noise, so
# they add little signal and just make the model harder to train.
INFORMATIVE_SENSORS = ["s2", "s3", "s4", "s7", "s11", "s12", "s15"]

# Threshold (in cycles) below which we call the engine "anomalous" / degrading.
ANOMALY_RUL_THRESHOLD = 30


def main():
    # =====================================================================
    # STEP 1 — Load the raw file
    # =====================================================================
    # The file is whitespace-separated with NO header row. There are 26
    # columns in a fixed order: engine_id, cycle, 3 operational settings,
    # then 21 sensor readings (s1..s21). We name them explicitly so the
    # rest of the script can refer to columns by meaningful names.
    #
    # sep=r"\s+" splits on any run of whitespace, which also gracefully
    # handles the trailing spaces at the end of each line in this dataset.
    column_names = (
        ["engine_id", "cycle", "setting1", "setting2", "setting3"]
        + [f"s{i}" for i in range(1, 22)]
    )
    df = pd.read_csv(RAW_PATH, sep=r"\s+", header=None, names=column_names)

    # =====================================================================
    # STEP 2 — Compute Remaining Useful Life (RUL) per row
    # =====================================================================
    # For each engine, its maximum observed cycle is the moment of failure.
    # RUL at any given cycle = (that engine's failure cycle) - (current cycle).
    # So the failure row gets RUL = 0 and every earlier row counts down to it.
    #
    # groupby("engine_id")["cycle"].transform("max") broadcasts each engine's
    # max cycle back onto every one of that engine's rows, so the subtraction
    # is fully vectorized (no Python loop over engines).
    max_cycle_per_engine = df.groupby("engine_id")["cycle"].transform("max")
    df["RUL"] = max_cycle_per_engine - df["cycle"]

    # =====================================================================
    # STEP 3 — Binary "anomaly" label
    # =====================================================================
    # Label = 1 when the engine is within its final 30 cycles (RUL <= 30),
    # i.e. it is noticeably degrading and close to failure; 0 otherwise.
    #
    # Why 30 cycles is a reasonable threshold for FD001 specifically:
    #   - FD001 engines run for ~128 cycles on average (and as few as ~128
    #     to ~360), so 30 cycles is roughly the final ~15-25% of an engine's
    #     life — long enough to be a useful early warning, short enough that
    #     the sensor degradation signal is actually present and separable.
    #   - The damage-propagation model behind CMAPSS produces fault signatures
    #     that only become clearly observable late in life; before that the
    #     readings look "healthy", so labeling them anomalous would inject noise.
    #   - 30 is the conventional "piecewise-linear RUL" clip point widely used
    #     in CMAPSS literature, which makes results comparable to other work
    #     and is a defensible, well-documented choice in an interview.
    # We cast the boolean to int so the column is a clean 0/1 integer label.
    df["anomaly"] = (df["RUL"] <= ANOMALY_RUL_THRESHOLD).astype(int)

    # =====================================================================
    # STEP 4 — Keep only the 7 most informative sensors
    # =====================================================================
    # Drop the operational settings and the noisy/flat sensors. We keep the
    # identifiers, our derived targets (RUL, anomaly), and the 7 good sensors.
    keep_cols = ["engine_id", "cycle", "RUL", "anomaly"] + INFORMATIVE_SENSORS
    df = df[keep_cols]

    # =====================================================================
    # STEP 5 — Export the clean CSV
    # =====================================================================
    # Make sure the output directory exists, then write without the pandas
    # index so the CSV contains only our chosen columns.
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    df.to_csv(OUT_PATH, index=False)

    # =====================================================================
    # STEP 6 — Print a summary
    # =====================================================================
    total_rows = len(df)
    anomaly_rate = df["anomaly"].mean() * 100  # mean of a 0/1 column = fraction of 1s
    num_engines = df["engine_id"].nunique()

    print("CMAPSS FD001 preprocessing complete")
    print(f"  Output file      : {OUT_PATH}")
    print(f"  Total rows       : {total_rows}")
    print(f"  Unique engines   : {num_engines}")
    print(f"  Anomaly rate     : {anomaly_rate:.2f}%  (rows with RUL <= {ANOMALY_RUL_THRESHOLD})")


if __name__ == "__main__":
    main()
