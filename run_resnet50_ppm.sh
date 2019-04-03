#!/bin/bash

device=$1
stage=$2

[ -z $device ] && device=cpu
[ -z $stage ] && stage=0

model=resnet50_ppm
data_dir=data/comp5421_TASK2
batch_size=16
epochs=10
init_lr=1e-2
mean="86.95358172 106.59307037 105.14808181"
tensor_size="224 224"
mirror=true
resized=true

./scripts/run.sh \
    --device $device \
    --stage $stage \
    --model $model \
    --data-dir $data_dir \
    --batch-size $batch_size \
    --epochs $epochs \
    --init-lr $init_lr \
    --mean "$mean" \
    --tensor-size "$tensor_size" \
    --mirror $mirror \
    --resized $resized
