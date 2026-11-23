"""
scripts/make_partition_table.py - Table 1 of the v2 manuscript (test windows and test prevalence by station),
computed from the archived test partition, plus the sizes of the validation and test partitions.

Inputs (tracked):  results/test_meta.csv                     station, month and label of every test window
                   results/json/results_exp28_orig_probs.npz  validation and test labels of the core models
Output:            results/json/partition_counts.json

The training partition (2,332 windows, 1938-1996) is not recomputed here: it needs the monthly panel built by
code/etl_v2.py from the AEMET daily records, which are not redistributed.
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, "results")
OUT_JSON = os.path.join(RES, "json", "partition_counts.json")


def main():
    meta = pd.read_csv(os.path.join(RES, "test_meta.csv"))
    meta["year"] = pd.to_datetime(meta["fecha"]).dt.year
    labels = np.load(os.path.join(RES, "json", "results_exp28_orig_probs.npz"))
    assert np.array_equal(labels["y_te"], meta["y_te"].to_numpy()), "test labels differ between archives"

    table = {}
    for station, g in meta.groupby("ciudad"):
        table[str(station).replace("_", " ")] = {
            "test": int(len(g)),
            "test_pos": int(g["y_te"].sum()),
            "test_rate": round(float(g["y_te"].mean()), 4),
            "last_test_year": int(g["year"].max()),
        }
    table["Total"] = {
        "val": int(labels["y_va"].size),
        "val_pos": int(labels["y_va"].sum()),
        "test": int(len(meta)),
        "test_pos": int(meta["y_te"].sum()),
        "test_rate": round(float(meta["y_te"].mean()), 4),
        "test_years": int(meta["year"].nunique()),
        "first_test_year": int(meta["year"].min()),
        "last_test_year": int(meta["year"].max()),
    }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)

    print(f"{'Station':<12} {'Test':>5} {'Last year':>10}   Test positive rate")
    print("-" * 52)
    for station, row in table.items():
        if station == "Total":
            continue
        print(f"{station:<12} {row['test']:>5} {row['last_test_year']:>10}   "
              f"{100 * row['test_rate']:.1f}% ({row['test_pos']} / {row['test']})")
    t = table["Total"]
    print("-" * 52)
    print(f"{'Total':<12} {t['test']:>5} {'':>10}   {100 * t['test_rate']:.1f}% ({t['test_pos']} / {t['test']})")
    print(f"validation windows: {t['val']}; test windows: {t['test']} in {t['test_years']} years "
          f"({t['first_test_year']}-{t['last_test_year']})")
    print(f"Saved partition counts to {OUT_JSON}")


if __name__ == "__main__":
    main()
