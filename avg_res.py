import argparse
import pandas as pd
import numpy as np

argparse_parser = argparse.ArgumentParser(
    prog="Compute averages from results",
    description="Compute averages from results.",
)

argparse_parser.add_argument("filename", help="Results filename.")
args = argparse_parser.parse_args()

import pandas as pd

# data = {
#     'prompt_tuning_02182025153915_rte_text_instruct_origin_0_meta-llama-3.1-8b-instruct_best': {
#         'test/accuracy': 0.9064748201438849, 'test/f1': 0.8925619834710744
#     },
#     'prompt_tuning_02182025195335_rte_text_instruct_origin_1_meta-llama-3.1-8b-instruct_best': {
#         'test/accuracy': 0.9064748201438849, 'test/f1': 0.8925619834710744
#     },
#     'prompt_tuning_02182025195335_rte_text_instruct_origin_2_meta-llama-3.1-8b-instruct_best': {
#         'test/accuracy': 0.8992805755395683, 'test/f1': 0.8870967741935484
#     }
# }

# # Convert to DataFrame
# df = pd.DataFrame.from_dict(data, orient='index')

# # Compute mean and std
# mean_values = df.mean()
# std_values = df.std()

# print(data)

# print(df, mean_values, std_values)

df = pd.read_csv(args.filename)

for pt in df["prompt_tuning"]:
    pt_df = pd.DataFrame.from_dict(eval(pt), orient="index")
    pt_df["test/accuracy"] = pd.to_numeric(pt_df["test/accuracy"])
    pt_df["test/f1"] = pd.to_numeric(pt_df["test/f1"])
    print(pt_df)

    mean_values = pt_df.mean()
    std_values = pt_df.std()
    print("mean:", np.round(mean_values["test/f1"]*100, 2))
    print("std:", np.round(std_values["test/f1"]*100, 2))