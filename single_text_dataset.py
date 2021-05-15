import torch
from torch.utils.data import Dataset
from typing import List


class SingleTextDataset(Dataset):
    def __init__(self, filenames: List[str], features: int = 10, targets: int = 1, max_size: int = -1):
        """
        Args :
            filenames : List of filenames to load data from
            features : Number of input features (characters)
            targets : Number of output features (characters)
            max_size : Set the maximum number of characters to read from file.  Defaults
            to -1 which is to read everything.
        """
        feature_list, target_list = dataset_from_file(
            filenames[0], features=features, targets=targets, max_size=max_size)
        self.inputs = feature_list
        self.output = target_list
        self.features = features
        self.targets = targets

    def __len__(self):
        return len(self.output)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        return (self.inputs[idx]-128+0.5)/128.0, self.output[idx]


def ascii_to_float(ascii_tensor: torch.Tensor):
    return (ascii_tensor-128+0.5)/128


def float_to_ascii(float_tensor: torch.Tensor):
    return ((float_tensor+1.0)*128-0.5).int()


def generate_dataset(text_in: str, features: int, targets: int):

    udata = text_in  # text_in.decode("utf-8")
    text = udata.encode("ascii", "ignore").decode('ascii')
    print('text[1:100', text[1:100])
    final = len(text)-(targets+features)
    feature_list = []
    target_list = []
    for i in range(final):
        n_feature = [ord(val) for val in text[i:(i+features)]]
        feature_list.append(n_feature)
        n_target = [ord(val)
                    for val in text[(i+features):(i+features+targets)]]
        target_list.append(n_target)

    return torch.tensor(feature_list), torch.tensor(target_list)


def dataset_from_file(filename: str, features: int, targets: int, max_size: int = -1):
    with open(filename, "r") as f:
        return generate_dataset(text_in=f.read()[0:max_size], features=features, targets=targets)
