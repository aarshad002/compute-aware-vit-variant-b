"""metrics.json saving and loading."""

import json
import os
from typing import Any, Dict


def save_metrics(metrics: Dict[str, Any], path: str) -> None:
    """Serialise a metrics dictionary to JSON.

    Creates parent directories if they do not exist.

    Args:
        metrics: Dictionary of metrics to persist.
        path: Absolute path to the output JSON file.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"Metrics saved → {path}")


def load_metrics(path: str) -> Dict[str, Any]:
    """Load a metrics dictionary from JSON.

    Args:
        path: Absolute path to the metrics JSON file.

    Returns:
        Parsed metrics dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Metrics file not found: {path}")
    with open(path, 'r') as f:
        return json.load(f)
