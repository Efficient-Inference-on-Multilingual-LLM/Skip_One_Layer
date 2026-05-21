import torch
import torch.nn as nn

from lm_eval.api.registry import register_model
from lm_eval.models.vllm_causallms import VLLM


class IdentityLayer(nn.Module):
    def forward(self, positions, hidden_states, residual=None, **kwargs):
        return (hidden_states, None) if residual is None else (hidden_states, residual)


def _apply_skips(model, indices):
    for i in indices:
        model.model.layers[i] = IdentityLayer()


@register_model("vllm_steered")
class VLLMSteeredModel(VLLM):
    def __init__(self, pretrained, steer_path, **kwargs):
        kwargs.setdefault("enforce_eager", True)
        super().__init__(pretrained=pretrained, **kwargs)

        steer_config = torch.load(steer_path, weights_only=False)
        skip_indices = [
            int(k.split(".")[-1])
            for k, v in steer_config.items()
            if v.get("action") == "skip"
        ]
        self.model.apply_model(lambda m: _apply_skips(m, skip_indices))
