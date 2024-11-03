from re import L

import numpy
import torch


def logarithmic_encoding(x: torch.Tensor, embedding_size: int = 256) -> torch.Tensor:
    """
    Encoding of a vector of float numbers converging them to its logarithm.

    To correspond to other embeddings it can be expended to any size, by copying the number itself.

    Example:
    >>> x = torch.Tensor([1, 2, 3])
    >>> logarithmic_encoding(x, 2)
    tensor([[0.0000, 0.0000],
            [0.6931, 0.6931],
            [1.0986, 1.0986]])

    Parameters
    ----------
    x : torch.FloatTensor
        torch tensor of shape N with numbers as float or int.
    embedding_size : int, optional
        Size of embedding dimention, by default 256

    Returns
    -------
    torch.Tensor
    """
    return torch.ones((1, embedding_size)) * torch.log(x.unsqueeze(1))


def identity_encoding(x: torch.Tensor, embedding_size: int = 256) -> torch.Tensor:
    """
    Identity encoding of a vector of float numbers.

    To correspond to other embeddings it can be expended to any size, by copying the number itself.

    Example:
    >>> x = torch.Tensor([1, 2, 3])
    >>> identity_encoding(x, 2)
    tensor([[1., 1.],
            [2., 2.],
            [3., 3.]])

    Parameters
    ----------
    x : torch.FloatTensor
        torch tensor of shape N with numbers as float or int.
    embedding_size : int, optional
        Size of embedding dimention, by default 256

    Returns
    -------
    torch.Tensor
    """
    return torch.ones((1, embedding_size)) * x.unsqueeze(1)


def sinusoidal_encoding(x: torch.Tensor, embedding_size: int = 256) -> torch.Tensor:
    """
    Sinusoidal encoding from original transformers paper.

    Example:
    >>> x = torch.Tensor([1, 2, 3])
    >>> sinusoidal_encoding(x, 2)
    tensor([[ 0.8415,  0.5403],
            [ 0.9093, -0.4161],
            [ 0.1411, -0.9900]])

    Parameters
    ----------
    x : torch.FloatTensor
        torch tensor of shape N with numbers as float or int.
    embedding_size : int, optional
        Size of embedding dimention, by default 256

    Returns
    -------
    torch.Tensor
    """
    log_inverse_10000 = -9.210340371976184
    arange_repeat_twice = (torch.arange(0, embedding_size) // 2 * 2).unsqueeze(0)
    log_division_term = log_inverse_10000 * (arange_repeat_twice / embedding_size)
    sinusoid_arguments = x.unsqueeze(1) * torch.exp(log_division_term)
    output = torch.empty_like(sinusoid_arguments)
    output[:, ::2] = torch.sin(sinusoid_arguments[:, ::2])
    output[:, 1::2] = torch.cos(sinusoid_arguments[:, 1::2])
    return output


def log_scaled_binary_encoding(
    x: torch.Tensor, embedding_size: int = 256
) -> torch.Tensor:
    """
    Binary representation of a nubmer multiplied by log of its original value.

    Example:
    >>> x = torch.Tensor([1, 2, 2.5, 3, 4, 5, 6])
    >>> log_scaled_binary_encoding(x, 3)
    tensor([[0.0000, 0.0000, 0.0000],
        [0.0000, 0.6931, 0.0000],
        [0.0000, 0.9163, 0.0000],
        [1.0986, 1.0986, 0.0000],
        [0.0000, 0.0000, 1.3863],
        [1.6094, 0.0000, 1.6094],
        [0.0000, 1.7918, 1.7918]])

    Parameters
    ----------
    x : torch.FloatTensor
        torch tensor of shape N with numbers as float or int.
    embedding_size : int, optional
        Size of embedding dimention, by default 256

    Returns
    -------
    torch.Tensor
    """
    x_as_long = torch.floor(x).int()
    log_of_x = torch.log(x + 2)
    return torch.stack(
        [
            torch.fmod(torch.bitwise_right_shift(x_as_long, bit_position), 2) * log_of_x
            for bit_position in range(embedding_size)
        ],
        dim=1,
    )
