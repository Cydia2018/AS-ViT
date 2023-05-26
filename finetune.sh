now=$(date +"%Y%m%d_%H%M%S")
logdir=train_log/exp_$now
datapath="/home/ssd3/dataset/imagenet/"

echo "output dir: $logdir"

python3 -m torch.distributed.launch --nproc_per_node=4 --master_port 23479 --use_env \
	main.py \
	--arch deit_small \
	--base_rate 0.65 \
	--input-size 224 \
	--sched cosine \
    --lr 2e-5 \
	--min-lr 2e-6 \
	--weight-decay 1e-6 \
	--batch-size 128 \
	--warmup-epochs 0 \
	--epochs 30 \
	--distill \
	--dist-eval \
	--data-path $datapath \
	--output_dir $logdir
	

echo "output dir for the last exp: $logdir"\