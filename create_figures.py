import seaborn as sns
import pandas as pd

import matplotlib.pyplot as plt

df = pd.read_csv("resuts.csv", index_col=0)
df = df.groupby(["tasks"], as_index=False).first() # comment when fixed

res_dict = {}

for t in df["tasks"]:
    print(t)
    for tt in t.split(" "):
        if t not in res_dict:
            res_dict[t] = {}
        
        res_dict[t].update({tt: df[df["tasks"] == t][f"{tt}_accuracy"].values[0]/df[df["tasks"] == tt][f"{tt}_accuracy"].values[0]})

print(res_dict)

data = [[],[]]

for t in res_dict:
    if len(t.split(" ")) == 2:
        for i, tt in enumerate(res_dict[t]):
            data[i].append(res_dict[t][tt])
print(data)

sns.scatterplot(data=res_dict)
plt.show()