#!/bin/bash

device=$1
stage=$2

[ -z $device ] && stage=cpu
[ -z $stage ] && stage=0

# parameters begins (change your parameters here)

model=zero_nn
data_dir=data/comp5421_TASK2
batch_size=32
epochs=1
init_lr=1e-2
mean="128 128 128"
tensor_size="321 321"
mirror=true
resized=true

# parameters end

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
    --resized $resized \
