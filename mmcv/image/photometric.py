import cv2
import numpy as np


def imnormalize(img, mean, std, to_rgb=True):
    """Normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    img = img.copy().astype(np.float32)
    return imnormalize_(img, mean, std, to_rgb)


def imnormalize_(img, mean, std, to_rgb=True):
    """Inplace normalize an image with mean and std.

    Args:
        img (ndarray): Image to be normalized.
        mean (ndarray): The mean to be used for normalize.
        std (ndarray): The std to be used for normalize.
        to_rgb (bool): Whether to convert to rgb.

    Returns:
        ndarray: The normalized image.
    """
    # cv2 inplace normalization does not accept uint8
    assert img.dtype != np.uint8
    mean = np.float64(mean.reshape(1, -1))
    stdinv = 1 / np.float64(std.reshape(1, -1))
    if to_rgb:
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)  # inplace
    cv2.subtract(img, mean, img)  # inplace
    cv2.multiply(img, stdinv, img)  # inplace
    return img


def imdenormalize(img, mean, std, to_bgr=True):
    assert img.dtype != np.uint8
    mean = mean.reshape(1, -1).astype(np.float64)
    std = std.reshape(1, -1).astype(np.float64)
    img = cv2.multiply(img, std)  # make a copy
    cv2.add(img, mean, img)  # inplace
    if to_bgr:
        cv2.cvtColor(img, cv2.COLOR_RGB2BGR, img)  # inplace
    return img


def iminvert(img):
    """Invert (negate) an image.

    Args:
        img (ndarray): Image to be inverted.

    Returns:
        ndarray: The inverted image.
    """
    return np.full_like(img, 255) - img


def solarize(img, thr=128):
    """Solarize an image (invert all pixel values above a threshold)

    Args:
        img (ndarray): Image to be solarized.
        thr (int): Threshold for solarizing (0 - 255).

    Returns:
        ndarray: The solarized image.
    """
    img = np.where(img < thr, img, 255 - img)
    return img


def posterize(img, bits):
    """Posterize an image (reduce the number of bits for each color channel)

    Args:
        img (ndarray): Image to be posterized.
        bits (int): Number of bits (1 to 8) to use for posterizing.

    Returns:
        ndarray: The posterized image.
    """
    shift = 8 - bits
    img = np.left_shift(np.right_shift(img, shift), shift)
    return img


def equalize(img):
    """Equalize the image histogram.

    This function applies a non-linear mapping to the input image,
    in order to create a uniform distribution of grayscale values
    in the output image.

    Args:
        img (ndarray): Image to be equalized.

    Returns:
        ndarray: The equalized image.
    """

    def _scale_channel(im, c, eps=1e-5):
        """Scale the data in the corresponding channel."""
        im = im[:, :, c]
        # Compute the histogram of the image channel.
        histo = np.histogram(im, 256, (0, 255))[0]
        # For computing the step, filter out the nonzeros.
        nonzero_histo = histo[histo > 0]
        step = (np.sum(nonzero_histo) - nonzero_histo[-1]) // 255

        # Build lut from the full histogram and step.
        # Compute the cumulative sum, shifted by step // 2
        # and then normalized by step.
        lut = (np.cumsum(histo) + (step // 2)) // max(step, eps)
        # Shift lut, prepending with 0.
        lut = np.concatenate([[0], lut[:-1]], 0)
        if step:
            # Clip the counts to be in range.
            lut = np.clip(lut, 0, 255)
        # If step is zero, return the original image.
        # Otherwise, index from lut.
        return np.where(np.equal(step, 0), im, lut[im])

    # Scales each channel independently and then stacks
    # the result.
    s1 = _scale_channel(img, 0)
    s2 = _scale_channel(img, 1)
    s3 = _scale_channel(img, 2)
    equalized_img = np.stack([s1, s2, s3], axis=-1)
    return equalized_img
