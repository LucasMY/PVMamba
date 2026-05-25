import os

from torch import nn
from mmengine.model import BaseModule
from mmseg.registry import MODELS as MODELS_MMSEG


def import_abspy(name="models", path="classification/"):
    import sys
    import importlib

    path = os.path.abspath(path)
    assert os.path.isdir(path), f"Invalid path: {path}"

    sys.path.insert(0, path)
    try:
        module = importlib.import_module(name)
    finally:
        sys.path.pop(0)
    return module


build = import_abspy(
    "models",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "../classification/"),
)

Backbone_PVMAMBA: nn.Module = build.Backbone_PVMAMBA


@MODELS_MMSEG.register_module()
class MM_PVMAMBA(BaseModule, Backbone_PVMAMBA):
    def __init__(self, *args, **kwargs):
        BaseModule.__init__(self)
        Backbone_PVMAMBA.__init__(self, *args, **kwargs)
        