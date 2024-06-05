#!/bin/bash

for i in nli cls sent
do
    python eval_cross_task_init.py ./configs/cross_task_init/full/$i.toml
done