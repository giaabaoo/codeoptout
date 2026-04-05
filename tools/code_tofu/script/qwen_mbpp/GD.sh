export CUDA_VISIBLE_DEVICES=5
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
master_port=18759
split=code50
model=gwen2.5-coder-7b-instruct
data_path=mbpp
lr=1.5e-6
loss=grad_diff
model_path=/cm/shared/baodhg2/Research/bao/Code_Copyright_Takedown/checkpoints/gwen2.5-coder-7b-instruct-finetune-mbpp

num_epochs=1

torchrun --nproc_per_node=1 --master_port=$master_port forget.py --config-name=forget_code.yaml forget_loss=${loss} \
 split=${split} batch_size=1 gradient_accumulation_steps=4 model_family=${model} data_path=${data_path} model_path=${model_path} lr=${lr} num_epochs=${num_epochs}
