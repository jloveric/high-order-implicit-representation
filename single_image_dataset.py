#from PIL import Image
from matplotlib import image
import torch
import numpy as np
from hilbertcurve.hilbertcurve import HilbertCurve
import math


def image_to_dataset(filename: str, peano: str = False):
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
    print('image.shape', torch_image.shape)
    max_size = max(torch_image.shape[0], torch_image.shape[1])

    xv, yv = torch.meshgrid(
        [torch.arange(torch_image.shape[0]), torch.arange(torch_image.shape[1])])

    # rescale so the maximum values is between -1 and 1
    xv = (xv/max_size)*2-1
    yv = (yv/max_size)*2-1

    xv = xv.reshape(xv.shape[0], xv.shape[1], 1)
    yv = yv.reshape(yv.shape[0], yv.shape[1], 1)

    torch_position = torch.cat([xv, yv], dim=2)
    torch_position = torch_position.reshape(-1, 2)
    
    if peano is True:
        # can index 2^{n*p} cubes with p = 2 (dimension)
        n = 2  # number of dimensions
        p = math.ceil(math.log(max_size, n)/2.0)
        hilbert_curve = HilbertCurve(p=p, n=2)
        cartesian_position = torch_position.tolist()
        hilbert_distances = hilbert_curve.distance_from_points(cartesian_position)
        

    torch_image_flat = torch_image.reshape(-1, 3)*2.0/255.0-1
    print('torch_max', torch.max(torch_image_flat))

    return torch_image_flat, torch_position, torch_image


if __name__ == "__main__":
    image_to_dataset(filename="images/newt.jpg")
