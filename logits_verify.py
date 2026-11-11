import torch
from download_weights import model
ids = torch.tensor([[15496, 11, 616, 1438, 318]])
with torch.no_grad():
    logits = model(ids).logits[0, -1]
print(f"{logits[0]:.6f} {logits[-1]:.6f} {logits.sum(dtype=torch.float64):.6f}")