#!/bin/bash

for i in {0,50,100,250,500,750,1000}
do
    python eval_cross_task_init.py ./configs/cross_task_init/fewshot/$1/$i.toml
done