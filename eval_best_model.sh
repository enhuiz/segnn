#!/bin/bash

device=$1

[ -z $device ] && device=cpu

./run_dilated_resnet18_upernet_refine.sh $device 1
