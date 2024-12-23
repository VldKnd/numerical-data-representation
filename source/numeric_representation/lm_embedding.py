import torch
from sentence_transformers import SentenceTransformer

from source.numeric_representation import BaseNumericModel


class MinilmEmbedding(torch.nn.Module, BaseNumericModel):
    """
    Simple embedding with small lanuage model
    """

    def __init__(self) -> None:
        super().__init__()
        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")

    def encode(self, input: list[int | float | str]) -> torch.Tensor:
        """
        Encodes contextual and numerical information of input

        Parameters
        ----------
        input : list[int  |  float  |  str]
            Input to be encoded.

        Returns
        -------
        torch.Tensor
        """
        contextual_embeddings = self.extract_contexctual_embeddings(input)

        return contextual_embeddings

    def extract_contexctual_embeddings(
        self, input: list[int | float | str]
    ) -> torch.Tensor:
        """
        Encodes contextual information of input

        Parameters
        ----------
        input : list[int  |  float  |  str]
            Input to be encoded.

        Returns
        -------
        torch.Tensor
        """
        input_as_string = [str(sentence).lower() for sentence in input]
        embedding_as_numpy_array = self.sentence_transformer.encode(input_as_string)
        return torch.from_numpy(embedding_as_numpy_array)
