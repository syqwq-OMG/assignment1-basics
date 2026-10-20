import copy

import numpy as np
import pytest
import torch

from cs336_basics.main import configure_runtime, evaluate_full, forward_logits, language_model_loss
from cs336_basics.Trans4mer.transformer import TransformerLM
from cs336_basics.Trans4mer.operat0r import softmax


def tiny_model():
    torch.manual_seed(17)
    return TransformerLM(64, 16, 8, 1, 2, 32, 10000)


def test_compiled_model_preserves_checkpoint_and_gradients():
    eager = tiny_model()
    compiled = copy.deepcopy(eager)
    configure_runtime(compiled, "fp32", True, compile_backend="aot_eager")
    inputs = torch.arange(16).reshape(2, 8) % 64
    targets = (inputs + 1) % 64
    losses = []
    for model in (eager, compiled):
        loss = language_model_loss(forward_logits(model, inputs), targets)
        loss.backward()
        losses.append(loss)
    torch.testing.assert_close(losses[0], losses[1])
    assert eager.state_dict().keys() == compiled.state_dict().keys()
    for a, b in zip(eager.parameters(), compiled.parameters()):
        torch.testing.assert_close(a.grad, b.grad)
    restored = tiny_model()
    restored.load_state_dict(compiled.state_dict(), strict=True)
    torch.testing.assert_close(restored(inputs), eager(inputs))


@pytest.mark.parametrize("compiled", [False, True])
def test_bf16_partial_full_validation_and_fp32_loss(compiled):
    model = configure_runtime(tiny_model(), "bf16", compiled, compile_backend="aot_eager")
    data = np.arange(22, dtype=np.uint16) % 64
    nll, count = evaluate_full(model, data, 2, 8, torch.device("cpu"))
    assert count == 21 and np.isfinite(nll)
    inputs = torch.arange(8).reshape(1, 8)
    logits = forward_logits(model, inputs)
    loss = language_model_loss(logits, inputs)
    assert loss.dtype == torch.float32
    loss.backward()
    assert all(p.dtype == torch.float32 and torch.isfinite(p.grad).all() for p in model.parameters())


def test_bf16_softmax_uses_float32_reduction():
    x = torch.linspace(-12, 12, 4096).to(torch.bfloat16)
    actual = softmax(x)
    expected = softmax(x.float()).to(torch.bfloat16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
