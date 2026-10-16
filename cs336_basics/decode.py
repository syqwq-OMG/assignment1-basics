from collections.abc import Sequence
import torch
from cs336_basics.Trans4mer.transformer import TransformerLM


def sample_next_token(
    logits: torch.Tensor, temperature: float = 1.0, top_p: float = 1.0, generator: torch.Generator | None = None
) -> int:
    if temperature < 0:
        raise ValueError("temperature cannot be negative")
    if not 0 < top_p <= 1:
        raise ValueError("top_p must satisfy 0 < top_p <= 1")

    if temperature == 0:
        return int(torch.argmax(logits).item())

    probabilities = torch.softmax(logits / temperature, dim=-1)
    if top_p < 1:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative_probabilities = torch.cumsum(sorted_probabilities, dim=-1)

        remove = cumulative_probabilities > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_probabilities.masked_fill_(remove, 0)
        sorted_probabilities /= sorted_probabilities.sum()
        sampled_position = torch.multinomial(sorted_probabilities, 1, generator=generator)
        return int(sorted_indices[sampled_position].item())

    return int(torch.multinomial(probabilities, 1, generator=generator).item())


@torch.inference_mode()
def decode(
    model: TransformerLM,
    prompt_tokens: Sequence[int],
    max_new_tokens: int,
    context_length: int,
    temperature: float = 1.0,
    top_p: float = 1.0,
    end_token_id: int | None = None,
    generator: torch.Generator | None = None,
) -> list[int]:
    device = next(model.parameters()).device
    generated = [int(token) for token in prompt_tokens]
    model.eval()
    for _ in range(max_new_tokens):
        model_input = torch.tensor(generated[-context_length:], dtype=torch.long, device=device).unsqueeze(0)
        next_token = sample_next_token(model(model_input)[0, -1], temperature, top_p, generator)
        generated.append(next_token)
        if end_token_id is not None and next_token == end_token_id:
            break
    return generated
