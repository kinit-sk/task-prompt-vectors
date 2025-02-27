#!/bin/bash

# for i in {0..9}; do mkdir -pv soft_prompts/origin_$i; done
# for i in {0..9}; do cp -v saves/origin_$i/origin_$i.bin soft_prompts/origin_$i; done

for i in {0..9}; do mkdir -pv soft_prompts/origin_"$i"_meta-llama-3.1-8b-instruct; done
for i in {0..9}; do cp -v saves/origin_"$i"_meta-llama-3.1-8b-instruct/origin_"$i"_meta-llama-3.1-8b-instruct.bin soft_prompts/origin_"$i"_meta-llama-3.1-8b-instruct; done

# datasets=("qnli_text" "mnli_text" "dbpedia_text" "sst2_text" "yelp_polarity_text" "trec_coarse_text")
# datasets=("qnli_text_instruct" "trec_coarse_text_instruct" "sst2_text_instruct", "yelp_polarity_text_instruct", "dbpedia_text_instruct")
# datasets=("squad_v2_instruct")
# datasets=("rte_text_instruct" "mrpc_text_instruct" "cola_text_instruct" "qqp_text_instruct" "stsb_text_instruct")
datasets=("rte_text" "mrpc_text" "cola_text" "qqp_text" "stsb_text")

for i in ${datasets[@]}; do for j in {0..2}; do occurances=(saves/*${i}*origin_${j}_t5-base_best); python sf2bin.py ${occurances[-1]} soft_prompts/origin_$j/$i.bin; done; done
# for i in ${datasets[@]}; do for j in {0..2}; do occurances=(saves/*${i}*origin_${j}_meta-llama-3.1-8b-instruct_best); python sf2bin.py ${occurances[-1]} soft_prompts/origin_"$j"_meta-llama-3.1-8b-instruct/$i.bin; done; done

# for i in ${datasets[@]}; do for j in {0..2}; do occurances=(saves/*${i}*origin_${j}_meta-llama-3.1-8b-instruct_best); echo ${occurances[-1]} soft_prompts/origin_"$j"_meta-llama-3.1-8b-instruct/$i.bin ; done; done