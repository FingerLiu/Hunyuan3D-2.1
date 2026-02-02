export CUDA_VISIBLE_DEVICES=1,2,3,4
export num_gpu_per_node=4
# export CUDA_VISIBLE_DEVICES=0
# export num_gpu_per_node=1

export node_num=1
export node_rank=0
export master_ip=0.0.0.0 # set your master_ip

# export config=configs/hunyuandit-finetuning-flowmatching-dinol518-bf16-lr1e5-4096.yaml
# export output_dir=output_folder/dit/fintuning_lr1e5

# export config=configs/hunyuandit-mini-overfitting-flowmatching-dinol518-bf16-lr1e4-4096.yaml
# export output_dir=output_folder/dit/overfitting_depth_16_token_4096_lr1e4
# 
# bash scripts/train_deepspeed.sh $node_num $node_rank $num_gpu_per_node $master_ip $config $output_dir

# 修改 train_demo.sh
export config=configs/hunyuandit-finetuning-4090-24gb-sem.yaml
export output_dir=output_folder/dit/finetuning_4090_sem_v3

echo "========== train_deepspeed 开始 =========="
echo "开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
start_sec=$(date +%s)

bash scripts/train_deepspeed.sh $node_num $node_rank $num_gpu_per_node $master_ip $config $output_dir
exit_code=$?

end_sec=$(date +%s)
elapsed=$((end_sec - start_sec))
elapsed_h=$((elapsed / 3600))
elapsed_m=$(((elapsed % 3600) / 60))
elapsed_s=$((elapsed % 60))

echo "========== train_deepspeed 结束 =========="
echo "结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
printf "总耗时: %d 小时 %d 分 %d 秒 (%d 秒)\n" $elapsed_h $elapsed_m $elapsed_s $elapsed
exit $exit_code
