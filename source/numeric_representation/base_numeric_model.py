from enum import Enum

import torch


class BaseNumericModel:
    """
    Base class to template common methods.
    """

    def __init__(self): ...

    def encode(self, input: list[int | float | str]) -> torch.Tensor:
        """
        Method that encodes input in embedding space.
        """
        ...


class EmbeddingClasses(Enum):
    """
    Accesible embedding types.
    """

    LANGUAGE_MODEL = "language_model"
    LOGARTIHMIC = "logarithmic"
    SIGMOID = "sigmoid"
    SINUSOIDAL = "sinusoidal"
