#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
source ./w-b.sh
python_bin="${SERVER_PYTHON:-/lmfs2026/env/a1/bin/python}"
data_dir="${ASSIGNMENT_DATA_DIR:-/ai/lmfs2026/data/assignment1-data}"
if [[ ! -x "$python_bin" ]]; then
    echo "Cannot execute $python_bin. Activate the course environment or set SERVER_PYTHON." >&2
    exit 1
fi
action="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$action" in
    check) exec "$python_bin" -m scripts.check_server --data-dir "$data_dir" "$@" ;;
    bpe)
        exec "$python_bin" record_bpe.py \
            --input-path "$data_dir/owt_train.txt" \
            --valid-path "$data_dir/owt_valid.txt" \
            --vocab-size 32000 "$@"
        ;;
    train) exec "$python_bin" transformer_main.py --device cuda \
        --train-text "$data_dir/owt_train.txt" --val-text "$data_dir/owt_valid.txt" "$@" ;;
    best) exec "$python_bin" -m scripts.train_best \
        --train-text "$data_dir/owt_train.txt" --val-text "$data_dir/owt_valid.txt" "$@" ;;
    *)
        echo 'Usage: bash run_server.sh check [--wandb]'
        echo '       bash run_server.sh bpe [--num-threads N]'
        echo '       bash run_server.sh train --tokenizer-prefix outputs/BPE_RUN_ID/bpe/tokenizer [training flags]'
        echo '       bash run_server.sh best --tokenizer-prefix outputs/BPE_RUN_ID/bpe/tokenizer [training flags]'
        ;;
esac
