#!/bin/bash

for i in {0,5,10,25,50,100,250,500,750,1000}
do
    python eval_cross_task_init.py ./configs/cross_task_init/fewshot/$1/$i.toml
done

# for i in {5,10,25}
# do
#     python eval_cross_task_init.py ./configs/cross_task_init/fewshot/$1/$i.toml
# done