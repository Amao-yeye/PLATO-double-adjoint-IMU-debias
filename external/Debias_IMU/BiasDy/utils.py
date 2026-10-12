import torch


def bmv(A, B):
    """
    Batch matrix vector multiplication
    :param A: shape (batch_size, m, n)
    :param B: shape (batch_size, n)
    :return: shape (batch_size, m)
    """
    return torch.matmul(A, B.unsqueeze(-1)).squeeze(-1)
