import numpy as np
from tqdm import tqdm

def histogram_stretching(img, h_min=0, h_max=1):
    """ Stretches the histogram of an image.

    Args:
        img: Image to stretch.
        h_min: Minimum value of the stretched image.
        h_max: Maximum value of the stretched image.

    Returns:
        Stretched image.
    """

    max_value = np.max(img)
    min_value = np.min(img)
    if max_value > 0 and min_value != max_value:
        return h_min+(h_max-h_min)*(img-min_value)/(max_value-min_value)
    else:
        return img


def painter(img1, img2, alpha2=0.5):
    """ Merges two images into one, according to alpha factor.

    Args:
        img1: 1st image
        img2: 2nd image
        alpha2: Importance of 2nd image. 1 maximum, 0 minimum.

    Returns:
        Merged images.
    """
    
    return (img1.astype('float') * (1 - alpha2)
            + img2.astype('float') * alpha2).astype('uint8')

def segments2colors(segments, image, kind='overlay', show_progress=False):
    """ Shows the segmentation on the input image. Works in the same way as
        label2rgb from skimage.color, either using random or average colors to
        fill the different regions.

    Args:
        segments (2d numpy array): segmentation of the image, of shape:
            [height, width], where each element represents the region
            (or segment) a pixel belongs to.
        image (3d numpy array): image that is being segmented.
        kind (str, optional): either 'overlay' to display a random color for
            each region or 'avg' to use the mean color of the region. When
            using 'overlay', the image is shown in the background, merging it
            with the segmentatino colors. Defaults to 'overlay'.
        show_progress (bool, optional): whether to show the progress of
            computing the colored image. Defaults to False.

    Returns:
        3d numpy array: image with the segmentation colors.
    """

    id_segments = np.unique(segments)
    colors = np.zeros(shape=segments.shape + (3,), dtype='uint8')
    progress = tqdm(id_segments) if show_progress else id_segments

    for id_seg in progress:
        mask = segments == id_seg

        if kind == 'overlay':
            colors[mask, :] = np.random.randint(0, 255, 3)
        elif kind == 'avg':
            colors[mask, :] = np.mean(image[mask, :], axis=0)

    if kind == 'overlay':
        return painter(colors, image)
    else:
        return colors
