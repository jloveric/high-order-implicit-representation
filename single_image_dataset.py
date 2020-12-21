#from PIL import Image
from matplotlib import image
import torch
import numpy as np


def image_to_dataset(filename: str):
    """
    Read in an image file and return the flattened position input
    flattened output and torch array of the original image.
    Args :
        filename : image filename.
    Returns :
        flattened image [width*heigh, rgb], flattened position vectory
        [width*height, 2] and torch tensor of original image.
    """
    img = image.imread(filename)

    torch_image = torch.from_numpy(np.array(img))
    xv, yv = torch.meshgrid(
        [torch.arange(torch_image.shape[0]), torch.arange(torch_image.shape[1])])
    xv = xv.reshape(xv.shape[0], xv.shape[1], 1)
    yv = yv.reshape(yv.shape[0], yv.shape[1], 1)

    torch_position = torch.cat([xv, yv], dim=2)
    torch_position = torch_position.reshape(-1, 2)

    torch_image_flat = torch_image.reshape(-1, 3)

    return torch_image_flat, torch_position, torch_image


if __name__ == "__main__":
    image_to_dataset("images/newt.jpg")
