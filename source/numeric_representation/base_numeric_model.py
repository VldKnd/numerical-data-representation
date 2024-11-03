from enum import Enum

import torch


class BaseNumericModel:
    def __init__(self): ...

    def encode(self, input: list[int | float | str]) -> torch.Tensor: ...


class EmbeddingClasses(Enum):
    LANGUAGE_MODEL = "language_model"
    LOGARTIHMIC = "logarithmic"
    SIGMOID = "sigmoid"
    SINUSOIDAL = "sinusoidal"
