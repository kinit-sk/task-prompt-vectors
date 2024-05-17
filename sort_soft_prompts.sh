#!/bin/bash

for i in {0..9}; do mkdir -pv soft_prompts/origin_$i; done
for i in {0..9}; do cp -v saves/origin_$i/origin_$i.bin soft_prompts/origin_$i; done

datasets=("qnli_text" "mnli_text" "dbpedia_text" "sst2_text" "yelp_polarity_text" "trec_coarse_text")
for i in ${datasets[@]}; do for j in {0..9}; do occurances=(saves/*${i}*origin_${j}_best); python sf2bin.py ${occurances[-1]} soft_prompts/origin_$j/$i.bin; done; done