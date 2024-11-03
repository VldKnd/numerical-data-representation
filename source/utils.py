import torch


def pairwise_cosine_similarity_matrix(x1, x2):
    """
    Compute pairwise cosine similarity matrix for two sets of vectors.

    Parameters
    ----------
    x1 : N x D
    x2 : M x D

    Returns
    -------
    torch.Tensor N x M
    """
    eps = 1e-10
    norm_x1, norm_x2 = x1.norm(dim=-1).unsqueeze(1), x2.norm(dim=-1).unsqueeze(1)
    x1_scaled = x1 / torch.max(norm_x1, eps * torch.ones_like(norm_x1))
    x2_scaled = x2 / torch.max(norm_x2, eps * torch.ones_like(norm_x2))
    cosine_similarity_matrix = torch.mm(x1_scaled, x2_scaled.transpose(0, 1))
    return cosine_similarity_matrix
