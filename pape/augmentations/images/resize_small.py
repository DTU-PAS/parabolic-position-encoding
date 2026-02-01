"""Adapted from https://github.com/GreenCUBIC/lookhere/blob/main/data_prep.py"""

from torchvision.transforms import v2 as T


class ResizeSmall(object):
    """Resizes the smaller side to `smaller_size` keeping aspect ratio.

    Args:
        smaller_size: an integer, that represents a new size of the smaller side of
        an input image.

    Returns:
        A function, that resizes an image and preserves its aspect ratio.
    """

    def __init__(self, smaller_size: int):
        self.smaller_size = smaller_size

    def __call__(self, image):
        h, w = image.shape[1], image.shape[2]  # image should be a tensor of shape (channels, height, width)

        # Figure out the necessary h/w.
        ratio = float(self.smaller_size) / min(h, w)
        new_h = int(h * ratio)
        new_w = int(w * ratio)
        image = T.Resize((new_h, new_w), antialias=True)(image)
        return image
