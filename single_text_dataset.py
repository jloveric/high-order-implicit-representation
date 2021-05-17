import torch
from torch.utils.data import Dataset
from typing import List, Tuple


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

        return (self.inputs[idx]-64+0.5)/64.0, self.output[idx]


def ascii_to_float(ascii_tensor: torch.Tensor):
    return (ascii_tensor-64+0.5)/64


def float_to_ascii(float_tensor: torch.Tensor):
    return ((float_tensor+1.0)*64-0.5).int()


def encode_input_from_text(text_in: str, features: int) -> Tuple[torch.tensor, str]:
    """
    Convert a string to input that the network can take.  Take the last "features" number
    of characters and convert to numbers.  Return those numbers as the network input, also
    return the raw_features (the text used to create the numbers).
    Args :
        text_in : input string.
        features : number of input features.
    Returns :
        tensor encoding, text used to create encoding.
    """
    text = text_in.encode("ascii", "ignore").decode('ascii')
    raw_sample = text[-(features):]
    encoding = [ord(val) for val in raw_sample]
    return torch.tensor(encoding), raw_sample


def decode_output_to_text(encoding: torch.tensor, topk: int = 1) -> Tuple[torch.tensor, str]:
    """
    Takes an output from the network and converts to text.
    Args :
        encoding : Tensor of size 128 for each ascii character
        topk : The number of maximum values to report back
    Returns :
        Tuple of topk values and corresponding topk indices and list containing
        actual ascii values.
    """
    probabilities = torch.nn.Softmax(dim=0)(encoding)

    ascii_codes = torch.topk(probabilities, k=topk, dim=0)
    ascii_values = [chr(val).encode("ascii", "ignore").decode('ascii')
                    for val in ascii_codes[1]]

    return ascii_codes[0], ascii_codes[1], ascii_values


def generate_dataset(text_in: str, features: int, targets: int):
    text = text_in.encode("ascii", "ignore").decode('ascii')
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
