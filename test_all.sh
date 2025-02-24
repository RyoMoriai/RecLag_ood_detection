#!/bin/bash  

for dataset in 'cifar10'
do
    for score in 'MSP' 'Energy' 'ReAct' 'HE' 'SHE'
    do
        for model in 'resnet18' 'resnet34'
        do
        python test_score_ood_detection.py --dataset $dataset --model $model --score $score --stored_data_path data --resize_val 112
        done
    python test_score_ood_detection.py --dataset $dataset --model wrn --score $score --stored_data_path data --resize_val 64
    done
done


