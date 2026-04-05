export CUDA_VISIBLE_DEVICES=4,5
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
master_port=18777
split=code50
model=gwen2.5-coder-7b-instruct
data_path=mbpp
lr=1.5e-6
loss=KL
model_path=/cm/shared/trucctt/code/codeoptout/checkpoints/gwen2.5-coder-7b-instruct-finetune-mbpp

num_epochs=1

GPUS_PER_NODE=$(python -c "import torch; print(torch.cuda.device_count());")
MASTER_ADDR=${MASTER_ADDR:-localhost}
NNODES=${WORLD_SIZE:-1}
NODE_RANK=${RANK:-0}
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
BATCH_SIZE=32
MICRO_BATCH_SIZE=4
GRAD_ACCU=$(($BATCH_SIZE / $WORLD_SIZE / $MICRO_BATCH_SIZE))

torchrun --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --node_rank $NODE_RANK \
    --master_addr $MASTER_ADDR \
    --master_port=$master_port forget.py --config-name=forget_code.yaml forget_loss=${loss} \
    split=${split} batch_size=${MICRO_BATCH_SIZE} gradient_accumulation_steps=${GRAD_ACCU} model_family=${model} data_path=${data_path} model_path=${model_path} lr=${lr} num_epochs=${num_epochs}
