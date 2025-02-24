#!/bin/bash  

for dataset in 'cifar10'
do
    for model in'resnet34'
    do
        python ./generate_stored_pattern.py --model $model --dataset $dataset --score 'HE'
    done
done

for dataset in 'cifar10' 
do
    for model in  'resnet34'
    do
        python ./generate_stored_pattern.py --model $model --dataset $dataset --score 'SHE'
    done
done