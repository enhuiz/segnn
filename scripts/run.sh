#!/bin/bash
device=cpu
stage=0

data_dir=data/comp5421_TASK2
model=
batch_size=32
epochs=8
init_lr=1e-2
mean="128 128 128"
tensor_size="321 321"
mirror=true
resized=true

. ./scripts/parse_options.sh

out_dir=exp/$model/

if [ $stage -le 0 ]; then
    model_path=zoo/$model.pth
    echo "$0: Training ..."
    python3 -u scripts/nn_train.py \
        --data-dir $data_dir/train \
        --out-dir $out_dir/train \
        --model-path $model_path \
        --device $device \
        --epochs $epochs \
        --batch-size $batch_size \
        --init-lr $init_lr \
        --tensor-size $tensor_size \
        --mean $mean \
        --mirror $mirror \
        --resized $resized
fi

if [ $stage -le 1 ]; then
    # the latest ckpt
    model_path=$(ls -1v $out_dir/train/ckpt/*.pth | tail -1)

    for task in val test; do
        echo "$0: Forward $task using $model_path ..."
        python3 -u scripts/nn_forward.py\
            --data-dir $data_dir/$task \
            --out-dir $out_dir/$task \
            --model-path $model_path \
            --device $device \
            --batch-size 1 \
            --tensor-size $tensor_size \
            --mean $mean
    done
fi

if [ $stage -le 2 ]; then
    echo "$0: Evaluating ..."
    pred_dir=$(readlink -f $out_dir/val/)
    cd $data_dir
    python3 -u eval.py --pred-dir $pred_dir
    cd -
fi
