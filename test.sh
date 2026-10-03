# uv run pytest -q tests/test_train_bpe.py
# uv run pytest -q tests/test_tokenizer.py

uv run pytest -k "linear \
    or embedding \
    or rmsnorm \
    or silu or swiglu \
    or softmax \
    or test_rope \
    or test_scaled_dot_product_attention \
    or test_multihead_self_attention \
    or run_transformer_block" 

# uv run python -m cs336_basics.BPE.bpe