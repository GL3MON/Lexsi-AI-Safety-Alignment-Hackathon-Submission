"""Memory-bounded output projection for the pinned TRL chunked loss."""


def projection_chunk(h, w, b, lbl, logit_scale, final_logit_softcapping):
    """Project in the head's dtype; compute CE/entropy in FP32.

    TRL 1.7.1 explicitly materializes w.float() for every chunk. Tiny Aya's
    262,144 x 2,048 head makes that a temporary 2 GiB copy. Ordinary native-dtype
    linear projection avoids the copy while retaining FP32 log probabilities.
    The output head is frozen in our LoRA configuration.
    """
    import torch
    from torch.nn import functional as F

    logits = F.linear(h.to(w.dtype), w, b).float()
    if logit_scale != 1.0:
        logits = logits * logit_scale
    if final_logit_softcapping is not None:
        logits = final_logit_softcapping * torch.tanh(logits / final_logit_softcapping)
    log_p = F.log_softmax(logits, dim=-1)
    loss = F.nll_loss(log_p, lbl, reduction="sum")
    correct = (logits.argmax(dim=-1) == lbl).sum().float()
    entropy = -(log_p.exp() * log_p).sum(dim=-1).sum()
    return loss, correct, entropy


def configure_chunked_loss(chunk_size, projection_dtype):
    from importlib.metadata import version
    from trl.trainer import sft_trainer

    if version("trl") != "1.7.1":
        raise RuntimeError("The local chunked-loss integration requires pinned TRL 1.7.1")
    if not hasattr(sft_trainer, "_tau2_original_chunk"):
        sft_trainer._tau2_original_chunk = sft_trainer._chunk
    sft_trainer._CHUNKED_LM_HEAD_CHUNK_SIZE = chunk_size
    # Scoped to this process; no installed package files are modified.
    sft_trainer._chunk = projection_chunk if projection_dtype == "native" else sft_trainer._tau2_original_chunk


def preserve_generation_signature(model):
    """Retain native generation capabilities after TRL wraps the model forward.

    GenerationMixin validates kwargs and detects logits_to_keep by inspecting
    forward's signature. TRL's chunked loss erases those named arguments even
    though its labels-free path delegates them to the unchanged native forward.
    """
    import inspect

    base = model.get_base_model()
    function = getattr(base.forward, "__func__", None)
    if function is not None and function.__name__ == "_chunked_ce_forward":
        function.__signature__ = inspect.signature(base.__class__.forward)
