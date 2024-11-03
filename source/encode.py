from functools import lru_cache

import torch

from source.numeric_representation import (
    BaseNumericModel,
    EmbeddingClasses,
    LoagrithmicMinilmEmbedding,
    MinilmEmbedding,
    SigmoidMinilmEmbedding,
    SinusoidalMinilmEmbedding,
)

EMBEDDING_TYPE_TO_EMBEDDING_CLASS: dict[EmbeddingClasses, type[BaseNumericModel]] = {
    EmbeddingClasses.LANGUAGE_MODEL: MinilmEmbedding,
    EmbeddingClasses.SINUSOIDAL: SinusoidalMinilmEmbedding,
    EmbeddingClasses.LOGARTIHMIC: LoagrithmicMinilmEmbedding,
    EmbeddingClasses.SIGMOID: SigmoidMinilmEmbedding,
}


@lru_cache
def get_embedding_model(embedding_type: str) -> BaseNumericModel:
    """
    Create embedding model from accessible list of classes.

    Parameters
    ----------
    embedding_type : str
        Type of embedding models.
        Has to one of 'language_model', 'logarithmic', 'sigmoid', 'sinusoidal'.

    Returns
    -------
    BaseNumericModel
        Instance of embedding model

    Raises
    ------
    RuntimeError
        If embedding type is not supported it will raise a runtime error.
    """
    try:
        embedding_type_as_enum = EmbeddingClasses(embedding_type)
        return EMBEDDING_TYPE_TO_EMBEDDING_CLASS[embedding_type_as_enum]()
    except ValueError:
        raise RuntimeError(
            (
                f"{embedding_type}  is not a valid  embedding_type, "
                f"please choose one of {[element.value for element in EmbeddingClasses]}."
            )
        )


def encode_number(
    input: int | str | float, embedding_type: str = "sinusoidal"
) -> torch.Tensor:
    """
    Encodes input with embeding from accessible list of classes.

    Parameters
    ----------
    input: int | str | float
        Number to be encoded.

    embedding_type : str
        Type of embedding models.
        Has to one of 'language_model', 'logarithmic', 'sigmoid', 'sinusoidal'.

    Returns
    -------
    torch.Tensor
        Encoding of a number

    Raises
    ------
    RuntimeError
        If embedding type is not supported it will raise a runtime error.
    """
    return encode_numbers([input], embedding_type=embedding_type)[0]


def encode_numbers(
    input: list[int | str | float], embedding_type: str = "sinusoidal"
) -> torch.Tensor:
    """
    Encodes list of inputs with embeding from accessible list of classes.

    Parameters
    ----------
    input: list[int | str | float]
        Inputs to be encoded.

    embedding_type : str
        Type of embedding models.
        Has to one of 'language_model', 'logarithmic', 'sigmoid', 'sinusoidal'.

    Returns
    -------
    torch.Tensor
        Encoding of a number

    Raises
    ------
    RuntimeError
        If embedding type is not supported it will raise a runtime error.
    """
    embedding_model = get_embedding_model(embedding_type)
    return embedding_model.encode(input)
