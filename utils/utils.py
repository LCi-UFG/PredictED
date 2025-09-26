import os, random
import numpy as np
import torch


def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


device = torch.device(
    "cuda" if torch.cuda.is_available() 
           else "cpu"
           )


def one_hot(value, categories):
    return [1 if i == categories.index(value) 
            else 0 for i in range(len(categories))
           ]


def decode_one_hot(one_hot_vector, categories):
    if isinstance(one_hot_vector, list):
        index = one_hot_vector.index(1)
    else:
        index = one_hot_vector.argmax().item()
    return categories[index]


def clip_gradients(
    model, 
    max_norm, 
    norm_type=2, 
    method='norm'):
    
    parameters = [p for p in model.parameters() 
                  if p.grad is not None
                 ]
    if method == 'norm':
        total_norm = torch.nn.utils.clip_grad_norm_(
            parameters, 
            max_norm, 
            norm_type=norm_type
            )
        return total_norm
    elif method == 'value':
        torch.nn.utils.clip_grad_value_(
            parameters, max_norm
            )
        return None
    else:
        raise ValueError(
            f"Unsupported clipping method: {method}"
            )