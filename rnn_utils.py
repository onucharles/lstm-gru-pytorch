import torch

def param_count(matrix):
    """Count the number of weights in a matrix or TT matrix"""
    assert isinstance(matrix, torch.nn.Module)
    total = 0
    for param in matrix.parameters():
        num = param.shape
        total += num.numel()
    
    return total