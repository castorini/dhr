import torch
import torch.nn as nn
from torch import Tensor

# remove_dim_dict = {768: -198, 640: -198, 512: 826, 256: 314, 128: 58}
# remove_dim_dict1 = {768: 570, 640: 442, 512: 314, 256: 58, 128: 58}

def cal_remove_dim(dims, vocab_size=30522):

    remove_dims = vocab_size % dims
    if remove_dims > 1000: # the first 1000 tokens in BERT are useless
        remove_dims -= dims

    return remove_dims

def aggregate(lexical_reps: Tensor,
            dims: int = 640, 
            remove_dims: int = -198, 
            full: bool = True
):

    if full:
        remove_dims = cal_remove_dim(dims*2)
        batch_size = lexical_reps.shape[0]
        if remove_dims >= 0:
            lexical_reps = lexical_reps[:, remove_dims:].view(batch_size, -1, dims*2)
        else:
            lexical_reps = torch.nn.functional.pad(lexical_reps, (0, -remove_dims), "constant", 0).view(batch_size, -1, dims*2)
        
        tok_reps, _ = lexical_reps.max(1)    

        positive_tok_reps = tok_reps[:, 0:2*dims:2]
        negative_tok_reps = tok_reps[:, 1:2*dims:2]

        positive_mask = positive_tok_reps > negative_tok_reps
        negative_mask = positive_tok_reps <= negative_tok_reps
        tok_reps = positive_tok_reps * positive_mask - negative_tok_reps * negative_mask
    else:
        remove_dims = cal_remove_dim(dims)
        batch_size = lexical_reps.shape[0]
        lexical_reps = lexical_reps[:, remove_dims:].view(batch_size, -1, dims)
        tok_reps, index_reps = lexical_reps.max(1)

    return tok_reps

