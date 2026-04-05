export CUDA_VISIBLE_DEVICES=4
export NCCL_P2P_DISABLE=1
export NCCL_DEBUG=INFO
export HYDRA_FULL_ERROR=1
master_port=18761
split=code50
model=deepseek6-7b
data_path=humaneval
lr=1.5e-6
loss=grad_ascent
model_path=/cm/shared/baodhg2/Research/bao/Code_Copyright_Takedown/checkpoints/deepseek-coder-6.7b-finetune-humaneval

num_epochs=3
torchrun --nproc_per_node=1 --master_port=$master_port forget.py --config-name=forget_code.yaml forget_loss=${loss} \
 split=${split} batch_size=1 gradient_accumulation_steps=4 model_family=${model} data_path=${data_path} model_path=${model_path} lr=${lr} num_epochs=${num_epochs}
