"""Small inference utilities."""

import torch


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def one_hot(value, categories):
    try:
        index = categories.index(value)
    except ValueError as error:
        raise ValueError(f"Unsupported molecular feature value: {value!r}") from error
    result = [0] * len(categories)
    result[index] = 1
    return result
