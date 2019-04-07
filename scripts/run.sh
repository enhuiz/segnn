#!/bin/bash
device=cpu
stage=0

data_dir=data/comp5421_TASK2
zoo_dir=zoo
model=
batch_size=32
epochs=8
init_lr=1e-2
mean="128 128 128"
input_height="400 300"
input_ratio="0.75 0.66"

. ./scripts/parse_options.sh

out_dir=exp/$model/

if [ $stage -le 0 ]; then
    model_path=$zoo_dir/$model.pth
    if [ ! -f $model_path ]; then
        python3 -u scripts/nn_make.py $zoo_dir || exit 1
        if [ ! -f $model_path ]; then
            echo "$0: $model_path not found, please implement your model in mknn.py"
            exit 1
        fi
    fi
    echo "$0: Training ..."
    python3 -u scripts/nn_train.py \
        --data-dir $data_dir/train \
        --out-dir $out_dir \
        --model-path $model_path \
        --device $device \
        --epochs $epochs \
        --input-height $input_height \
        --input-ratio $input_ratio \
        --batch-size $batch_size \
        --init-lr $init_lr \
        --mean $mean || exit 1
fi

forward() {
    local task=$1

    # the latest ckpt
    model_path=$(ls -1v $out_dir/ckpt/*.pth | tail -1)

    if [ -z $model_path ]; then
        echo "Error: No model has been trained."
        exit 1
    fi

    echo "$0: Forward $task using $model_path ..."
    python3 -u scripts/nn_forward.py \
        --data-dir $data_dir/$task \
        --out-dir $out_dir/$task \
        --model-path $model_path \
        --input-height $input_height \
        --input-ratio $input_ratio \
        --device $device \
        --mean $mean
}

if [ $stage -le 1 ]; then
    forward val
fi

if [ $stage -le 2 ]; then
    echo "$0: Evaluating ..."
    pred_dir=$(readlink -f $out_dir/val/)
    cd $data_dir
    python3 -u eval.py --pred-dir $pred_dir
    cd -
fi

if [ $stage -le 3 ]; then
    forward test
fi
