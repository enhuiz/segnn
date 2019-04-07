#!/bin/bash

device=$1
stage=$2

[ -z $device ] && device=cpu
[ -z $stage ] && stage=0

model=dilated_resnet18_upernet_refine # after 20 epochs of dilated_resnet18_upernet.sh
data_dir=data/comp5421_TASK2
batch_size=5
epochs=5
init_lr=1e-3
mean="86.95358172 106.59307037 105.14808181"
input_size="800 600"

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
