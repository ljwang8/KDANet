from pathlib import Path

import yaml


def load_config(path):
    path = Path(path)
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)
