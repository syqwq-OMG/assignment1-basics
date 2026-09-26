# uv run pytest -q tests/test_train_bpe.py
# uv run pytest -q tests/test_tokenizer.py

uv run pytest -k "linear or embedding or rmsnorm or swiglu" 

# uv run python -m cs336_basics.BPE.bpe