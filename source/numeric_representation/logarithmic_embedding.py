import torch
from sentence_transformers import SentenceTransformer

from source.numeric_representation import BaseNumericModel


class LoagrithmicMinilmEmbedding(torch.nn.Module, BaseNumericModel):
    def __init__(self) -> None:
        super().__init__()
        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_size = 384

    def encode(self, input: list[int | float | str]):
        numbers_from_inputs = torch.Tensor(
            [self.extract_number(sentence) for sentence in input]
        )
        numerical_embeddings = self.extract_numerical_embeddings(numbers_from_inputs)
        contextual_embeddings = self.extract_contexctual_embeddings(input)

        return numerical_embeddings * contextual_embeddings

    def extract_contexctual_embeddings(
        self, input: list[int | float | str]
    ) -> torch.Tensor:
        input_as_string = [str(sentence).lower() for sentence in input]
        embedding_as_numpy_array = self.sentence_transformer.encode(input_as_string)
        return torch.from_numpy(embedding_as_numpy_array)

    def extract_number(self, input: int | float | str) -> float:
        if type(input) is str:
            for possible_number in input.split(" "):
                try:
                    return float(possible_number)
                except ValueError:
                    continue
        else:
            return float(input)

        return -1

    def extract_numerical_embeddings(self, input: torch.Tensor) -> torch.Tensor:
        return torch.log(input + 1 + torch.e).unsqueeze(1)
