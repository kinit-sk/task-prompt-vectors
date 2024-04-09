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

data = {"first_task": [], "second_task": [], "tasks": []}

for t in res_dict:
    if len(t.split(" ")) == 2:
        first_task, second_task = t.split(" ")
        data["first_task"].append(res_dict[t][first_task])
        data["second_task"].append(res_dict[t][second_task])
        data["tasks"].append(t)

data = pd.DataFrame.from_dict(data)
print(data)

sns.scatterplot(data=data, x="first_task", y="second_task")
                
plt.vlines(x=1, ymin=0, ymax=1.5, colors='g',linestyles='dashed')
plt.hlines(y=1, xmin=0, xmax=1.5, colors='g',linestyles='dashed')

plt.show()