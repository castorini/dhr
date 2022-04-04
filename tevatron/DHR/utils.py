import torch
import torch.nn as nn
from torch import Tensor

def densify(lexical_reps: Tensor,
            dims: int = 768, 
            strategy: str = 'stride', 
            remove_dims: int = 570
):
    
    if not (len(lexical_reps.shape)==2):
        raise ValueError( 'Input lexical representation shape should be 2 (batch, vocab), but the input shape is {}'.format( len(lexical_reps.shape) ) )

    orig_dims = lexical_reps.shape[-1]
    if not ( (orig_dims-remove_dims)%dims==0 ):
        raise ValueError('Input lexical representation cannot be densified, please fix dims or remove_dims')

    # Todo: add other strategy
    batch_size = lexical_reps.shape[0]
    lexical_reps = lexical_reps[:, remove_dims:].view(batch_size, -1, dims)
    value_reps, index_reps = lexical_reps.max(1)    
    return value_reps, index_reps
