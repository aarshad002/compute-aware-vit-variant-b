"""FLOPs computation using fvcore."""

import torch
import torch.nn as nn

try:
    from fvcore.nn import FlopCountAnalysis
    _FVCORE_AVAILABLE = True
except ImportError:
    _FVCORE_AVAILABLE = False


def compute_flops(model: nn.Module, input_tensor: torch.Tensor) -> float:
    """Compute GFLOPs for one forward pass with the given input.

    For pruning models, input_tensor determines which tokens are kept;
    pass a representative sample (e.g. a zero tensor) to get a lower-bound
    or a real image for an accurate sample estimate.

    Falls back to 0.0 and prints a warning if fvcore is unavailable.

    Args:
        model: The model to profile.
        input_tensor: Example input of shape (1, 3, 224, 224).

    Returns:
        GFLOPs as a float.
    """
    if not _FVCORE_AVAILABLE:
        print("Warning: fvcore not installed — FLOPs reported as 0.")
        return 0.0

    was_training = model.training
    model.eval()
    try:
        flops = FlopCountAnalysis(model, input_tensor)
        flops.unsupported_ops_warnings(False)
        flops.uncalled_modules_warnings(False)
        result = flops.total() / 1e9
    except Exception as exc:
        print(f"Warning: FLOPs computation failed ({exc}) — reporting 0.")
        result = 0.0
    finally:
        model.train(was_training)
    return result
