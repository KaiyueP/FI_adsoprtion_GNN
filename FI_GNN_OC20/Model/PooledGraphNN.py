from abc import ABCMeta
from torch_geometric.nn import global_mean_pool, global_add_pool, global_max_pool
from typing import Literal
import torch

class PooledGraphNN(torch.nn.Module, metaclass=ABCMeta):
    """
    Acts as an abstract class for graph neural networks that
    scatter together their node values
    """

    def __init__(self, pooling: Literal["mean", "add", "max", "adsorbate_mean", "local_mean"]) -> None:
        """
        Initializes a scatter graph neural network with a self.pool function
        with the given mode

        Args:
            pooling: The mode of how to combine the nodes.
        """
        super().__init__()
        self.pooling = pooling
        if pooling in ("mean", "adsorbate_mean", "local_mean"):
            self.pool = global_mean_pool
        elif pooling == "add":
            self.pool = global_add_pool
        elif pooling == "max":
            self.pool = global_max_pool
        else:
            raise Exception("Illegal pooling mode.")


def adsorbate_mean_pool(x, batch, mask):
    """Average selected node features per graph without changing message passing."""
    if mask is None or mask.ndim != 1 or mask.shape[0] != x.shape[0]:
        raise ValueError('Adsorbate pooling requires one boolean mask value per atom.')
    if mask.dtype != torch.bool:
        raise ValueError('adsorbate_mask must be boolean.')
    size = int(batch.max().item()) + 1
    counts = global_add_pool(mask.to(x.dtype).unsqueeze(-1), batch, size=size)
    if (counts == 0).any():
        raise ValueError('Every graph must contain at least one adsorbate atom.')
    return global_add_pool(x[mask], batch[mask], size=size) / counts
