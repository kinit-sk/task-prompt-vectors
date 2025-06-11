from arithmetics import TaskPrompt
import torch
from sklearn.metrics.pairwise import cosine_similarity

import seaborn as sns

import pandas as pd
import matplotlib.pyplot as plt

tasks = ["qnli", "mnli", "trec_coarse", "dbpedia", "sst2", "yelp_polarity"]
origins = ["origin"]


def get_task_prompts(origin_prompts, dataset_names):
    return {
        origin_prompt: [
            TaskPrompt(
                prompt_name,
                task_weights=torch.load(f"soft_prompts/{prompt_name}.bin")[
                    "prompt_embeddings"
                ],
                origin_weigts=torch.load(f"soft_prompts/{origin_prompt}.bin")[
                    "prompt_embeddings"
                ],
            )
            for prompt_name in dataset_names
        ]
        for origin_prompt in origin_prompts
    }


def get_tasks(dataset_names):
    return [
        torch.load(f"soft_prompts/{prompt_name}.bin")["prompt_embeddings"]
        for prompt_name in dataset_names
    ]


tp = get_task_prompts(origins, tasks)
prompt_vectors = []
task_weights_vectors = []

for p in tp:
    for t in tp[p]:
        prompt_vectors.append(torch.flatten(t.prompt))

task_weights = get_tasks(tasks)
for t in task_weights:
    task_weights_vectors.append(torch.flatten(t))

prompt_vectors = torch.stack(prompt_vectors)
task_weights_vectors = torch.stack(task_weights_vectors)

cs = cosine_similarity(prompt_vectors, prompt_vectors)
tcs = cosine_similarity(task_weights_vectors, task_weights_vectors)

print(cs)
print(tcs)

df = pd.DataFrame(cs)
df.columns = tasks
df.index = tasks

plt.figure(0)
heatmap = sns.heatmap(df, annot=True, fmt=".2f", cmap="crest")

fig = heatmap.get_figure()
fig.savefig("heatmap_task_prompts.png", bbox_inches="tight")

df = pd.DataFrame(tcs)
df.columns = tasks
df.index = tasks

plt.figure(1)
heatmap = sns.heatmap(df, annot=True, fmt=".2f", cmap="crest")

fig = heatmap.get_figure()
fig.savefig("heatmap_task_weights.png", bbox_inches="tight")
