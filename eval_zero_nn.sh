device=$1

model=zero_nn
mean="128 128 128"
tensor_size="321 321"
batch_size=1
model_path=exp/$model/ckpt/1.pth

for type in val test; do
    data_dir=data/comp5421_TASK2/$type
    out_dir=eval/$model/$type

    python3 -u scripts/nn_forward.py\
        --data-dir $data_dir \
        --model-path $model_path \
        --out-dir $out_dir \
        --device $device \
        --batch-size $batch_size \
        --tensor-size $tensor_size \
        --mean $mean
done
