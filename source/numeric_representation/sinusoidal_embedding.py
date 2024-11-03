import torch
from sentence_transformers import SentenceTransformer

from source.numeric_representation import BaseNumericModel


class SinusoidalMinilmEmbedding(torch.nn.Module, BaseNumericModel):
    def __init__(self) -> None:
        super().__init__()

        self.sentence_transformer = SentenceTransformer("all-MiniLM-L6-v2")
        self.embedding_size = 384

        # Compute coefficients for sinusoidal embedding
        log_inverse_10000 = -9.210340371976184
        arange_repeat_twice = (torch.arange(0, self.embedding_size) // 2 * 2).unsqueeze(
            0
        )
        log_division_term = log_inverse_10000 * (
            arange_repeat_twice / self.embedding_size
        )

        # Cache computation for sinusoidal embedding
        self.division_term = torch.exp(log_division_term)

    def encode(self, input: list[int | float | str]):
        numbers_from_inputs = torch.Tensor(
            [self.extract_number(sentence) for sentence in input]
        )

        numerical_embeddings = self.extract_numerical_embeddings(
            input=numbers_from_inputs
        )

        contextual_embeddings = self.extract_contexctual_embeddings(input)

        return (
            numerical_embeddings + contextual_embeddings
        )  # Follow transformer implementation

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

        return 0

    def extract_numerical_embeddings(self, input: torch.Tensor) -> torch.Tensor:
        """
        Sinusoidal encoding from original transformers paper.

        Example:
        >>> x = torch.Tensor([0, 1, 2])
        >>> sinusoidal_encoding(x) # embedding size is explicitly assumed to be 2
        tensor([[ 0.8415,  0.5403],
                [ 0.9093, -0.4161],
                [ 0.1411, -0.9900]])

        Parameters
        ----------
        x : torch.FloatTensor
            torch tensor of shape N with numbers as float or int.

        Returns
        -------
        torch.Tensor
        """
        sinusoid_arguments = input.unsqueeze(1) + 1 * self.division_term
        output = torch.empty_like(sinusoid_arguments)
        output[:, ::2] = torch.sin(sinusoid_arguments[:, ::2])
        output[:, 1::2] = torch.cos(sinusoid_arguments[:, 1::2])
        return output
