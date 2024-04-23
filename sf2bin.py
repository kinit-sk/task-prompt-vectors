from safetensors import safe_open
import torch
import sys

argv = sys.argv

tensors = {}
print(argv)
with safe_open(f"{argv[1]}/adapter_model.safetensors", framework="pt", device=0) as f:
    for k in f.keys():
        tensors[k] = f.get_tensor(k)

with open(argv[2], "wb") as f:
    torch.save(tensors["prompt_embeddings"], f)
