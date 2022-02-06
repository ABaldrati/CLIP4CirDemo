import json
from pathlib import Path
from typing import List, Optional

import PIL.Image
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

server_base_path = Path(__file__).absolute().parent.absolute()
data_path = Path(__file__).absolute().parent.absolute() / 'data'


def _convert_image_to_rgb(image):
    return image.convert("RGB")


class TargetPad:
    """
    Pad the image if its aspect ratio is above a target ratio.
    Pad it to match such target ratio
    """

    def __init__(self, target_ratio: float, size: int, pad_value: Optional[int] = 0):
        """
        :param target_ratio: target ratio
        :param size: preprocessing output dimension
        :param pad_value: padding value, 0 is black-pad (zero-pad), 255 is white-pad
        """
        self.size = size
        self.target_ratio = target_ratio
        self.pad_value = pad_value

    def __call__(self, image):
        w, h = image.size
        actual_ratio = max(w, h) / min(w, h)
        if actual_ratio < self.target_ratio:  # check if the ratio is above or below the target ratio
            return image
        scaled_max_wh = max(w, h) / self.target_ratio  # rescale the pad to match the target ratio
        hp = max(int((scaled_max_wh - w) / 2), 0)
        vp = max(int((scaled_max_wh - h) / 2), 0)
        padding = [hp, vp, hp, vp]
        return F.pad(image, padding, self.pad_value, 'constant')


def targetpad_resize(target_ratio: float, dim: int, pad_value: int):
    """
    Yield a torchvision transform which resize and center crop an image using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :param pad_value: padding value, 0 is black-pad (zero-pad), 255 is white-pad
    :return: torchvision Compose transform
    """
    return Compose([
        TargetPad(target_ratio, dim, pad_value),
        Resize(dim, interpolation=PIL.Image.BICUBIC),
        CenterCrop(dim),
    ])


def targetpad_transform(target_ratio: float, dim: int):
    """
    CLIP-like preprocessing transform computed after using TargetPad pad
    :param target_ratio: target ratio for TargetPad
    :param dim: image output dimension
    :return: CLIP-like torchvision Compose transform
    """
    resize = targetpad_resize(target_ratio, dim, 0)
    return Compose([
        resize,
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])


class FashionIQDataset(Dataset):
    """
    FashionIQ dataset class which manage FashionIQ data
    The dataset yield tuples made of (image_name, image)
    The dataset manage an arbitrary numbers of FashionIQ category, e.g. only dress, dress+toptee+shirt, dress+shirt...
    """

    def __init__(self, split: str, dress_types: List[str], preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'val']
        :param dress_types: list of fashionIQ category
        :param preprocess: function which preprocess the image
        """
        self.dress_types = dress_types
        self.split = split

        if split not in ['test', 'val']:
            raise ValueError("split should be in ['test', 'val']")
        for dress_type in dress_types:
            if dress_type not in ['dress', 'shirt', 'toptee']:
                raise ValueError("dress_type should be in ['dress', 'shirt', 'toptee']")

        self.preprocess = preprocess

        # get the image names
        self.image_names: list = []
        for dress_type in dress_types:
            with open(
                    server_base_path / 'fashionIQ_dataset' / 'image_splits' / f'split.{dress_type}.{split}.json') as f:
                self.image_names.extend(json.load(f))

        print(f"FashionIQ {split} - {dress_types} dataset initialized")

    def __getitem__(self, index):
        try:
            image_name = self.image_names[index]
            image_path = server_base_path / 'fashionIQ_dataset' / 'images' / f"{image_name}.jpg"
            image = self.preprocess(PIL.Image.open(image_path))
            return image_name, image

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        return len(self.image_names)


class CIRRDataset(Dataset):
    """
    CIRR dataset class which manage CIRR data
    The dataset can be used in 'relative' or 'classic' mode:
    This dataset yield tuples made of (image_name, image)
    """

    def __init__(self, split: str, preprocess: callable):
        """
        :param split: dataset split, should be in ['test', 'val']
        :param preprocess: function which preprocess the image
        """
        self.preprocess = preprocess
        self.split = split

        if split not in ['test1', 'val']:
            raise ValueError("split should be in ['test1', 'val']")

        # get a mapping from image name to relative path
        with open(server_base_path / 'cirr_dataset' / 'cirr' / 'image_splits' / f'split.rc2.{split}.json') as f:
            self.name_to_relpath = json.load(f)

        print(f"CIRR {split} dataset initialized")

    def __getitem__(self, index):
        try:
            image_name = list(self.name_to_relpath.keys())[index]
            image_path = server_base_path / 'cirr_dataset' / self.name_to_relpath[image_name]
            im = PIL.Image.open(image_path)
            image = self.preprocess(im)
            return image_name, image

        except Exception as e:
            print(f"Exception: {e}")

    def __len__(self):
        return len(self.name_to_relpath)

# ---------------------------------------------------------
