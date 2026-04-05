export CUDA_VISIBLE_DEVICES=4
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
export CUDA_LAUNCH_BLOCKING=1
master_port=18759
split=code50
model=qwen2.5-coder-7b-base
data_path=mbpp
lr=1.5e-6
loss=grad_ascent
model_path=/cm/shared/trucctt/code/codeoptout/checkpoints/gwen2.5-coder-7b-finetune-mbpp

num_epochs=1
torchrun --nproc_per_node=1 --master_port=$master_port forget.py --config-name=forget_code.yaml forget_loss=${loss} \
 split=${split} batch_size=4 gradient_accumulation_steps=4 model_family=${model} data_path=${data_path} model_path=${model_path} lr=${lr} num_epochs=${num_epochs}
