#! /bin/bash
ROOT_PATH=$(pwd)
exp_setting=$1
model_name=qwen

cd $ROOT_PATH/runtime

if [ "$exp_setting" == "small" ]; then
    #### Model info ####
    model_size=0_6B

    #### Hardware info ####
    NNODES=1
    GPUS_PER_NODE=4
    WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

    #### Distributed info ####
    NODE_RANK=0
    MASTER_ADDR=localhost
    MASTER_PORT=7000
    DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

    #### Paths ####
    RESULT_PATH=${ROOT_PATH}/logs/aceso/
    LOG_PATH=${RESULT_PATH}runtime/${model_name}/${model_size}/
    CONFIG_SAVE_PATH=${RESULT_PATH}configs/${model_name}/${model_size}/top_configs/
    mkdir -p ${LOG_PATH}csv

    for file_name in $(ls $CONFIG_SAVE_PATH)
    do
    config_name=`basename $file_name .json`
    CURRENT_TIME=$(date '+%Y-%m-%d-%H-%M-%S')
    echo "[LOG][RUNTIME]($(date '+%Y-%m-%d-%H-%M-%S')) start executing config: $config_name ." >> ${RESULT_PATH}full_log.log

    # Use Qwen-specific pretrain entry (pretrain_qwen.py)
    # If local tokenizer exists, pass vocab/merge files; else rely on defaults inside pretrain_qwen
    # TOKENIZER_ARGS=""
    # if [ -f "vocabs/qwen/vocab.json" ] && [ -f "vocabs/qwen/merges.txt" ]; then
    #     TOKENIZER_ARGS="--tokenizer-type GPT2BPETokenizer --vocab-file vocabs/qwen/vocab.json --merge-file vocabs/qwen/merges.txt"
    # fi
    # --flexpipe-config $CONFIG_SAVE_PATH${file_name} \
    # --train-iters 3 \
    # --eval-iters 0 \
    # --distributed-backend nccl \
    # --log-path $LOG_PATH \
    # $TOKENIZER_ARGS \
    python3 -m torch.distributed.launch $DISTRIBUTED_ARGS \
        pretrain_qwen.py \
        --flexpipe-config $CONFIG_SAVE_PATH${file_name} \
        --train-iters 3 \
        --eval-iters 0 \
        --lr-decay-iters 320000 \
        --tokenizer-path model_configs/qwen3-0.6B \
        --vocab-file vocabs/qwen3-0.6b-vocab.json \
        --merge-file vocabs/qwen3-0.6b-merges.txt \
        --data-impl mmap \
        --split 949,50,1 \
        --distributed-backend nccl \
        --lr 0.00015 \
        --lr-decay-style cosine \
        --min-lr 1.0e-5 \
        --weight-decay 1e-2 \
        --clip-grad 1.0 \
        --lr-warmup-fraction .01 \
        --log-interval 1 \
        --DDP-impl local \
        --fp16 \
        --log-path $LOG_PATH \
        2>&1 | tee ${LOG_PATH}full_log_${config_name}_rank${NODE_RANK}_${CURRENT_TIME}

    echo "[LOG][RUNTIME]($(date '+%Y-%m-%d-%H-%M-%S')) end executing config: $config_name ." >> ${RESULT_PATH}full_log.log

    done

else
    echo "Only 'small' setting currently supported by this wrapper."
fi

python3 scripts/show_best_perf.py $model_name $RESULT_PATH 2>&1 | tee -a ${RESULT_PATH}full_log.log

