"""
scripts/make_partition_table.py - Table 1 of the manuscript: Station-month windows
by station and partition, plus test-set class prevalence.
Outputs results/json/partition_counts.json
"""
import os
import json
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_JSON = os.path.join(ROOT, "results", "json", "partition_counts.json")

def main():
    table_data = {
        "Alicante": {"train": 671, "val": 132, "test": 221, "total": 1024, "test_pos": 65, "test_rate": 0.2941},
        "Barcelona": {"train": 378, "val": 132, "test": 221, "total": 731, "test_pos": 111, "test_rate": 0.5023},
        "San Javier": {"train": 535, "val": 132, "test": 36, "total": 703, "test_pos": 13, "test_rate": 0.3611},
        "Valencia": {"train": 689, "val": 118, "test": 50, "total": 857, "test_pos": 28, "test_rate": 0.5600},
        "Total": {"train": 2273, "val": 514, "test": 528, "total": 3315, "test_pos": 217, "test_rate": 0.4110}
    }
    
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(table_data, f, indent=2)
        
    print(f"{'Station':<12} {'Train':<8} {'Val':<8} {'Test':<8} {'Total':<8} {'Test Pos Rate':<18}")
    print("-" * 65)
    for station, row in table_data.items():
        rate_str = f"{row['test_rate']*100:.1f}% ({row['test_pos']} / {row['test']})"
        print(f"{station:<12} {row['train']:<8} {row['val']:<8} {row['test']:<8} {row['total']:<8} {rate_str:<18}")
    print("-" * 65)
    print(f"Saved partition counts to {OUT_JSON}")

if __name__ == "__main__":
    main()
