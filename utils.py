from typing import Optional

import torch
from torch import nn
from torchinfo import summary


def collate_fn(batch: list):
    """
    Discard None images in a batch when using torch DataLoader
    :param batch: input_batch
    :return: output_batch = input_batch - None_values
    """
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_output_size(model: nn.Module, input_size: Optional[int] = None, preprocess: Optional[callable] = None) -> int:
    """
    Getting model output size given a known input or a CLIP-like preprocess pipeline
    :param model: CLIP input model
    :param input_size: input size on which calculate the output
    :param preprocess: CLIP-like preprocess pipeline
    :return: CLIP embedding dimension
    """
    if preprocess:
        transform_size = preprocess.transforms[0].size
        input_size = (1, 3, transform_size, transform_size)
    model_summary = summary(model, input_size=input_size, verbose=0)
    output_dim = model_summary.summary_list[-1].output_size[-1]
    return output_dim
