import importlib
from typing import Optional

import torch


def load_teacher(factory: str, checkpoint: Optional[str] = None, device: str = "cpu"):
    """Load a teacher model from a user-provided factory.

    The factory should use the form "module.submodule:function_name" and return
    a torch.nn.Module. The returned model must produce logits and feature maps.
    """

    if ":" not in factory:
        raise ValueError("Teacher factory must use the form 'module.submodule:function_name'.")

    module_name, function_name = factory.split(":", 1)
    module = importlib.import_module(module_name)
    model = getattr(module, function_name)()

    if checkpoint:
        state = torch.load(checkpoint, map_location=device)
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]
        model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    return model
