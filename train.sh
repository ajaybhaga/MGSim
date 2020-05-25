#!/bin/bash
#mpiexec -n $1 CUDA_VISIBLE_DEVICES=0 python3 DeepMimic_Optimizer.py --arg_file $2

#python3 testGPU.py
#mpiexec -n $1 python3 DeepMimic_Optimizer.py --arg_file $2
LD_LIBRARY_PATH=/home/nekokitty/dev/freeglut-3.2.1/build/lib/:/usr/local/lib/
export LD_LIBRARY_PATH

mpiexec -n $1 python3 MGSim_Optimizer.py --arg_file $2
