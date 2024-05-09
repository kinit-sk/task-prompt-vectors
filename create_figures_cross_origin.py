import pandas as pd

import glob

import argparse

argparse_parser = argparse.ArgumentParser(
    prog="Create averages from results",
    description="Compute average and std over all results.",
)

argparse_parser.add_argument("results_dir", help="Path to the results directory.")
argparse_parser.add_argument("--repeat", action="store_true",  help="Also include results with same origin comparison (repeating origins).")
argparse_parser.add_argument("--only", action="store_true", help="Only include same origin comparison.")
args = argparse_parser.parse_args()

dfs = []

for file in sorted(glob.glob(f"{args.results_dir}/results_origin*.csv")):
    print(file)
    filename_split = file.split("/")[-1].split("_")

    if not args.repeat and filename_split[2] == filename_split[4]:
        continue

    if args.only and filename_split[2] != filename_split[4]:
        continue

    print(filename_split[2], filename_split[4])

    df = pd.read_csv(file, index_col=0)

    df["tasks"] = df["tasks"].map(lambda x: " ".join(sorted(x.split(" "))))

    dfs.append(df)

mean_dfs = pd.concat(dfs).groupby("tasks", as_index=False).mean()
mean_dfs_std = pd.concat(dfs).groupby("tasks", as_index=False).std()

mean_dfs.to_csv(f"{args.results_dir}/average_10.csv")
mean_dfs_std.to_csv(f"{args.results_dir}/std_10.csv")