#!/bin/bash

device=$1
stage=$2

[ -z $device ] && device=cpu
[ -z $stage ] && stage=0

model=dilated_resnet18_upernet_refine
data_dir=data/comp5421_TASK2
batch_size=16
epochs=20
init_lr=5e-3
mean="86.95358172 106.59307037 105.14808181"
input_size="300 400" # more than half of the images (1146) are in 3:4


./scripts/run.sh \
    --device $device \
    --stage $stage \
    --model $model \
    --data-dir $data_dir \
    --batch-size $batch_size \
    --epochs $epochs \
    --init-lr $init_lr \
    --mean "$mean" \
    --input-size "$input_size" 
