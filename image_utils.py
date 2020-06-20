import os
import numpy as np
import skimage.filters as sk_filters
import skimage.exposure as sk_exposure

from PIL import Image, ImageDraw, ImageOps
from skimage import io, color
from pathlib import Path


def load_pil_image(path, gray=False, color_model="RGB"):

    with open(path, 'rb') as f:

        if gray:
            return Image.open(f).convert('L')     # grayscale

        elif color_model == "HSV":
            return Image.open(f).convert('HSV')      # hsv

        elif color_model == "LAB":
            rgb = io.imread(path)
            if rgb.shape[2] > 3:  # removes the alpha channel
                rgb = color.rgba2rgb(rgb)

            lab = color.rgb2lab(rgb)
            lab_scaled = ((lab + [0, 128, 128]) / [100, 255, 255])*255
            return Image.fromarray(lab_scaled.astype(np.uint8))

        return Image.open(f).convert('RGB')    # rgb


def save_pil_image(pil_image, filepath):

    if not os.path.exists(Path(filepath).parent):
        os.makedirs(Path(filepath).parent)

    pil_image.save(filepath)


def pil_to_np(pil_img):
    """
    Convert a PIL Image to a NumPy array.
    Note that RGB PIL (w, h) -> NumPy (h, w, 3).
    Args:
    pil_img: The PIL Image.
    Returns:
    The PIL image converted to a NumPy array.
    """

    rgb = np.asarray(pil_img)
    return rgb


def np_to_pil(np_img):
    """
    Convert a NumPy array to a PIL Image.
    Args:
        np_img: The image represented as a NumPy array.
    Returns:
    The NumPy array converted to a PIL Image.
    """

    if np_img.dtype == "bool":
        np_img = np_img.astype("uint8") * 255
    elif np_img.dtype == "float64":
        np_img = (np_img * 255).astype("uint8")

    return Image.fromarray(np_img)


def show_pil_img(pil_img):
    """
    Display a PIL image on the screen.
    Args:
        pil_img: image to show.
    """

    pil_img.show()


def show_np_img(np_img):
    """
    Convert a NumPy array to a PIL image, add text to the image, and display the image.
    Args:
        np_img: Image as a NumPy array.
    """

    result = np_to_pil(np_img)
    # if gray, convert to RGB for display
    if result.mode == 'L':
        result = result.convert('RGB')
    draw = ImageDraw.Draw(result)
    result.show()


def rgb_to_grayscale(np_img, output_type="uint8"):
    """
    Convert an RGB NumPy array to a grayscale NumPy array.
    Shape (h, w, c) to (h, w).
    Args:
        np_img: RGB Image as a NumPy array.
        output_type: Type of array to return (float or uint8)
    Returns:
        Grayscale image as NumPy array with shape (h, w).
    """

    grayscale = np.dot(np_img[..., :3], [0.2125, 0.7154, 0.0721])
    if output_type != "float":
        grayscale = grayscale.astype("uint8")

    return grayscale


def complement_np_img(np_img, output_type="uint8"):
    """
    Obtain the complement of an image as a NumPy array.
    Args:
        np_img: Image as a NumPy array.
        type: Type of array to return (float or uint8).
    Returns:
        Complement image as Numpy array.
    """

    if output_type == "float":
        complement = 1.0 - np_img
    else:
        complement = 255 - np_img

    return complement


def basic_threshold(np_img, threshold, output_type="bool"):
    """
    Return mask where a pixel has a value if it exceeds the threshold value.
    Args:
        np_img: Binary image as a NumPy array.
        threshold: The threshold value to exceed.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array representing a mask where a pixel has a value (T, 1.0, or 255) if the corresponding input array pixel exceeds the threshold value.
    """

    result = (np_img > threshold)
    if output_type == "bool":
        pass
    elif output_type == "float":
        result = result.astype(float)
    else:
        result = result.astype("uint8") * 255

    return result


def hysteresis_threshold(np_img, low=50, high=100, output_type="uint8"):
    """
    Apply two-level (hysteresis) threshold to an image as a NumPy array, returning a binary image.
    Args:
        np_img: Image as a NumPy array.
        low: Low threshold.
        high: High threshold.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above hysteresis threshold.
    """

    hyst = sk_filters.apply_hysteresis_threshold(np_img, low, high)
    if output_type == "bool":
        pass
    elif output_type == "float":
        hyst = hyst.astype(float)
    else:
        hyst = (255 * hyst).astype("uint8")

    return hyst


def otsu_threshold(np_img, output_type="uint8"):
    """
    Compute Otsu threshold on image as a NumPy array and return binary image based on pixels above threshold.
    Args:
        np_img: Image as a NumPy array.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a pixel above Otsu threshold.
    """

    otsu_thresh_value = sk_filters.threshold_otsu(np_img)
    otsu = (np_img > otsu_thresh_value)
    if output_type == "bool":
        pass
    elif output_type == "float":
        otsu = otsu.astype(float)
    else:
        otsu = otsu.astype("uint8") * 255

    return otsu


def contrast_stretch(np_img, low=40, high=60):
    """
    Filter image (gray or RGB) using contrast stretching to increase contrast in image based on the intensities in a specified range.
    Args:
        np_img: Image as a NumPy array (gray or RGB).
        low: Range low value (0 to 255).
        high: Range high value (0 to 255).
    Returns:
        Image as NumPy array with contrast enhanced.
    """

    low_p, high_p = np.percentile(np_img, (low * 100 / 255, high * 100 / 255))
    contrast_stretch = sk_exposure.rescale_intensity(np_img, in_range=(low_p, high_p))

    return contrast_stretch


def histogram_equalization(np_img, nbins=256, output_type="uint8"):
    """
    Filter image (gray or RGB) using histogram equalization to increase contrast in image.
    Args:
        np_img: Image as a NumPy array (gray or RGB).
        nbins: Number of histogram bins.
        output_type: Type of array to return (float or uint8).
    Returns:
        NumPy array (float or uint8) with contrast enhanced by histogram equalization.
    """

    # if uint8 type and nbins is specified, convert to float so that nbins can be a value besides 256
    if np_img.dtype == "uint8" and nbins != 256:
        np_img = np_img / 255
    hist_equ = sk_exposure.equalize_hist(np_img, nbins=nbins)
    if output_type == "float":
        pass
    else:
        hist_equ = (hist_equ * 255).astype("uint8")

    return hist_equ


def histogram_equalization_adaptive(np_img, nbins=256, clip_limit=0.01, output_type="uint8"):
    """
    Filter image (gray or RGB) using adaptive equalization to increase contrast in image, where contrast in local regions is enhanced.
    Args:
        np_img: Image as a NumPy array (gray or RGB).
        nbins: Number of histogram bins.
        clip_limit: Clipping limit where higher value increases contrast.
        output_type: Type of array to return (float or uint8).
    Returns:
        NumPy array (float or uint8) with contrast enhanced by adaptive equalization.
    """

    adapt_equ = sk_exposure.equalize_adapthist(np_img, nbins=nbins, clip_limit=clip_limit)
    if output_type == "float":
        pass
    else:
        adapt_equ = (adapt_equ * 255).astype("uint8")

    return adapt_equ


def entropy(np_img, neighborhood=9, threshold=5, output_type="uint8"):
    """
    Filter image based on entropy (complexity).
    Args:
        np_img: Image as a NumPy array.
        neighborhood: Neighborhood size (defines height and width of 2D array of 1's).
        threshold: Threshold value.
        output_type: Type of array to return (bool, float, or uint8).
    Returns:
        NumPy array (bool, float, or uint8) where True, 1.0, and 255 represent a measure of complexity.
    """

    entr = sk_filters.rank.entropy(np_img, np.ones((neighborhood, neighborhood))) > threshold
    if output_type == "bool":
        pass
    elif output_type == "float":
        entr = entr.astype(float)
    else:
        entr = entr.astype("uint8") * 255

    return entr


def cut_image_by_mask(image, mask, foreground='black', inverse=False):

    if inverse:
        mask = ImageOps.invert(mask)

    foreground = Image.new('RGB', image.size, color=foreground)
    return Image.composite(image, foreground, mask)