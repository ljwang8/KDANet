"""Example teacher adapter.

Copy this file and replace `build_teacher` with your HTransNet constructor.
The returned teacher must output either a dictionary with "logits" and
"features", or a tuple `(logits, features)`.
"""

from torch import nn


class ExampleTeacher(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("Replace this class with the HTransNet implementation.")

    def forward(self, x):
        logits = None
        features = None
        return {"logits": logits, "features": features}


def build_teacher():
    return ExampleTeacher()
