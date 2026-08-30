import sys
import pandas as pd

def summarize(path, time_column):
    df = pd.read_csv(path)

    times = pd.to_numeric(df[time_column], errors="coerce").dropna()

    print(f"\n{path}")
    print("-" * len(path))
    print(f"Count:    {len(times)}")
    print(f"Mean:     {times.mean():.6f}")
    print(f"Median:   {times.median():.6f}")
    print(f"Min:      {times.min():.6f}")
    print(f"Max:      {times.max():.6f}")
    print(f"Std dev:  {times.std():.6f}")
    print(f"P50:      {times.quantile(0.50):.6f}")
    print(f"P90:      {times.quantile(0.90):.6f}")
    print(f"P95:      {times.quantile(0.95):.6f}")
    print(f"P99:      {times.quantile(0.99):.6f}")

if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python compare_times.py RTX.csv L40S.csv TIME_COLUMN")
        sys.exit(1)

    _, rtx_csv, l40s_csv, time_column = sys.argv

    summarize(rtx_csv, time_column)
    summarize(l40s_csv, time_column)
