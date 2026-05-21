import torch
import torch.nn as nn
import lm_eval
from lm_eval.models.vllm_causallms import VLLM


class IdentityLayer(nn.Module):
    """Passthrough replacement for a transformer decoder layer."""
    def forward(self, positions, hidden_states, residual=None, **kwargs):
        # Match the (hidden_states, residual) tuple signature vLLM decoder
        # layers use in Llama/Mistral/Qwen2 — adjust per family if needed.
        if residual is None:
            return hidden_states, None
        return hidden_states, residual


def skip_layers(model, indices):
    layers = model.model.layers
    for i in indices:
        print(f"[skip] layer {i} -> Identity")
        layers[i] = IdentityLayer()


# 1) Build the harness VLLM wrapper — this internally constructs vllm.LLM
lm = VLLM(
    pretrained="Qwen/Qwen3-1.7B",
    dtype="bfloat16",
    gpu_memory_utilization=0.2,
    max_model_len=4096,
    batch_size="auto",
)

# 2) Reach into the vllm.LLM and patch layers
LAYERS_TO_SKIP = [10]  # last 8 of 32
lm.model.apply_model(lambda m: skip_layers(m, LAYERS_TO_SKIP))

# 3) Run the eval
results = lm_eval.simple_evaluate(
    model=lm,
    tasks=["mmlu", "hellaswag", "arc_challenge", "gsm8k"],
    num_fewshot=0,
    batch_size="auto",
)

print(results["results"])

# Save
import json, pathlib
pathlib.Path("results").mkdir(exist_ok=True)
with open(f"results/skip_{'-'.join(map(str, LAYERS_TO_SKIP))}.json", "w") as f:
    json.dump(results["results"], f, indent=2)