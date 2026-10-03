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
    or test_transformer_block \
    or test_transformer_lm \
    or test_cross_entropy" 

# uv run python -m cs336_basics.BPE.bpe