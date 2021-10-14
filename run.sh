#!/bin/bash

question=$1
train_data=$2
test_data=$3

if [[ ${question} == "1" ]]; then
part_num=$4
python3 Q1/q1.py $train_data $test_data $part_num
fi

if [[ ${question} == "2" ]]; then
multi=$4
part_num=$5
python3 Q2/q2.py $train_data $test_data $multi $part_num
fi