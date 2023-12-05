import json
import shutil
import os
from munch import Munch


def load_cfg(cfg_path="config.json"):
    assert os.path.exists(cfg_path), "config.json is missing!"
    with open(cfg_path, 'rb') as f:
        cfg = json.load(f)
    cfg = Munch(cfg)
    return cfg

