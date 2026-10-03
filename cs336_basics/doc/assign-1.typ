#import "lib.typ": *
#show: report.with(tit: [LMFS Assignment 1 Solution])


= Byte-Pair Encoding (BPE) Tokenizer

= Transformer

== CE Loss

Let $p_(theta) (dot mid(|) x_[i])$ be predicted distribution for the $i+1$ token, $o_(i) in RR^(B times V) = p_(theta) (dot mid(|) x_[i])$, the correct token probability is $o_i [x_(i+1)]$. For CE loss, we calculate the $-log$ of each correct token's probability and take the average over each next-token-prediction.

$
  "CE"(o, x) & = -frac(1, B) sum_(i) p_(theta) (x_(i+1) mid(|) x_([i])) \
             & = -frac(1, B) sum_(i) log frac(exp(o_i [x_(i+1)]), sum_(j) exp(o_i [j])) \
             & = frac(1, B) sum_(i) (log sum_(j) exp(o_i [j]) - o_i [x_(i+1)]) \
             & = frac(1, B) sum_(i) (log sum_(j) exp(o_i [j] -t) - o_i [x_(i+1)] +t), forall t
$

Code:
```py
def cross_entropy_loss(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute the cross-entropy loss between logits and targets.
    """
    inputs = inputs - torch.max(inputs, dim=-1, keepdim=True).values
    t = torch.log(torch.sum(torch.exp(inputs), dim=-1, keepdim=True)) - inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return torch.mean(t)

```
