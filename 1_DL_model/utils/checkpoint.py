from collections import OrderedDict

import torch

from core.unet import Unet


def _strip_module_prefix(state_dict):
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return OrderedDict(
        (key[7:] if key.startswith("module.") else key, value)
        for key, value in state_dict.items()
    )


def load_unet(checkpoint_path, device=None):
    """Load a Unet checkpoint for inference on CPU or CUDA."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("state_dict", checkpoint)

    model = Unet(1)
    model.load_state_dict(_strip_module_prefix(state_dict))
    model.to(device)
    model.eval()
    return model
