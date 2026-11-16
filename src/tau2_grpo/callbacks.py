"""Training callback that records real LoRA parameter updates."""

from transformers import TrainerCallback


class AdapterUpdateAudit(TrainerCallback):
    """Snapshot the actual initialized training adapter, not an earlier baseline model."""

    def __init__(self):
        self.before = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        self.before = {
            name: parameter.detach().float().cpu().clone()
            for name, parameter in model.named_parameters()
            if parameter.requires_grad
        }
        if not self.before or not all("lora_" in name for name in self.before):
            raise RuntimeError("Expected only LoRA parameters to be trainable.")
