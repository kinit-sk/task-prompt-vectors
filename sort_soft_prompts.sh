#!/bin/bash

for i in {0..9}; do mkdir -pv soft_prompts/origin_$i; done
for i in {0..9}; do cp -v saves/origin_$i/origin_$i.bin soft_prompts/origin_$i; done

datasets=("qnli" "mnli" "dbpedia" "sst2" "yelp_polarity")
for i in ${datasets[@]}; do for j in {0..9}; do python sf2bin.py saves/*${i}*origin_${j}_best soft_prompts/origin_$j/$i.bin; done; done
