device=$1

model=zero_nn
out_dir=exp/$model
model_path=zoo/$model.pth
data_dir=data/comp5421_TASK2/train
batch_size=32
epochs=8
init_lr=1e-2
mean="128 128 128"
tensor_size="321 321"
mirror=true
resized=true

python3 -u scripts/nn_train.py \
    --data-dir $data_dir \
    --model-path $model_path \
    --out-dir $out_dir \
    --device $device \
    --epochs $epochs \
    --batch-size $batch_size \
    --init-lr $init_lr \
    --tensor-size $tensor_size \
    --mean $mean \
    --mirror $mirror \
    --resized $resized
