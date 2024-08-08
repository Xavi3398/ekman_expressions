import os
import cv2
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.metrics import f1_score, precision_recall_fscore_support
import numpy as np

from ekman_expressions.utils import histogram_stretching

def save_heatmaps(img_accum, path_gray=None, path_colormap=None, colormap=cv2.COLORMAP_JET, min_th=None):
    """ Save heatmaps in gray scale and in a colormap.

    Args:
        img_accum: Accumulated image.
        path_gray: Path to save the gray scale heatmap.
        path_colormap: Path to save the heatmap in a colormap.
        colormap: Colormap to use.
        min_th: Minimum threshold to apply.
    """
    heatmap = img_accum
    heatmap = histogram_stretching(heatmap, 0, 1)
    if min_th is not None:
        heatmap[heatmap < min_th] = 0
    heatmap = (heatmap * 255).astype('uint8')

    # Save in gray scale
    if path_gray is not None:
        cv2.imwrite(path_gray, heatmap)

    # Save in a COLORMAP
    if path_colormap is not None:
        heatmap = cv2.applyColorMap(heatmap, colormap=colormap)
        cv2.imwrite(path_colormap, heatmap)

def apply_threshold(heatmap, th='otsu'):
    """ Apply a threshold to a heatmap.

    Args:
        heatmap: Heatmap to threshold.
        th: Threshold to apply. It can be 'otsu', an integer (if image is in [0,255]) 
            or a float (if image is in [0, 1]).

    Returns:
        Thresholded heatmap.
    """
    if th == 'otsu':
        _, th_heatmap = cv2.threshold(heatmap, 0, 255, cv2.THRESH_OTSU)
    elif type(th) == int:
        _, th_heatmap = cv2.threshold(heatmap, th, 255, cv2.THRESH_BINARY)
    elif type(th) == float:
        _, th_heatmap = cv2.threshold(heatmap, int(th*255), 255, cv2.THRESH_BINARY)
    else:
        raise Exception('Bad value for th parameter.')
    return th_heatmap

def distance_gray(image1, image2, method=cv2.TM_CCORR_NORMED):
    """ Compute distance between two gray images.

    Args:
        image1: 1st image.
        image2: 2nd image.
        method: Method to compute distance.

    Returns:
        Distance between the two images.
    """
    tm_res = cv2.matchTemplate(image1, image2, method)[0][0]
    return tm_res if method == cv2.TM_SQDIFF_NORMED else 1 - tm_res

def iou(image1, image2):
    """ Compute Intersection over Union between two binary images.

    Args:
        image1: 1st binary image.
        image2: 2nd binary image.

    Returns:
        Intersection over Union between the two images.
    """
    return np.sum(np.logical_and(image1, image2)) / np.sum(np.logical_or(image1, image2))

def accuracy(image1, image2):
    """ Compute accuracy between two binary images.

    Args:
        image1: 1st binary image.
        image2: 2nd binary image.

    Returns:
        Accuracy between the two images.
    """
    return np.sum(image1 == image2) / image1.size

def get_best_threshold(gt_image, heatmap, distance_metric='f1_score', range_white=[0.5, 1.5], path_save=None):
    """ Get the best threshold for a heatmap, given a distance metric.

    Args:
        gt_image: Ground truth image.
        heatmap: Heatmap to threshold.
        distance_metric: Distance metric to use. It can be 'f1_score', 'iou' or 'white'.
        range_white: Range of white pixels to consider.
        path_save: Path to save thresholded images.

    Returns:
        Best thresholded image.
    """

    # Lists of thresholded images and their f1_score
    ths = []
    scores = []

    # Ground truth and heatmap flattened
    gt = (gt_image/255).flatten()
    hm = heatmap

    total_white = np.sum(gt)
    min_white = total_white * range_white[0]
    max_white = total_white * range_white[1]

    # For each gray value
    for i in range(256):

        # Apply current threshold
        th = np.zeros_like(hm)
        th[hm > i] = 1

        sum_th = np.sum(th)

        if sum_th <= max_white:
            ths.append(th)

            # Compute distance with ground truth
            if distance_metric == 'f1_score':
                dist = f1_score(gt, th.flatten())
            elif distance_metric == 'iou':
                dist = iou(gt, th.flatten())
            elif distance_metric == 'white':
                dist = np.abs(sum_th - total_white)
            scores.append(dist)

            if path_save is not None:
                cv2.imwrite(os.path.join(path_save, str(i) + '_' + str(dist) + '.png'), th * 255)

            if sum_th < min_white:
                break

    # Return thresholded image with maximum f1_score
    if distance_metric == 'white':
        return ths[np.argmin(scores)]*255
    else:
        return ths[np.argmax(scores)]*255

def plot_dendogram(labels, heatmaps, title, save_path, distance_method=cv2.TM_CCOEFF_NORMED,
                   linkage_method='average', width=10, height=15, title_size=20, 
                   font_size=16, lims=None, leaf_rotation=45):
    """ Plot dendogram of heatmaps.

    Args:
        labels: Labels of heatmaps.
        heatmaps: Heatmaps to compare.
        title: Title of the plot.
        save_path: Path to save the plot.
        distance_method: Distance method to use.
        linkage_method: Linkage method to use.
        width: Width of the plot.
        height: Height of the plot.
        title_size: Size of the title.
        font_size: Size of the font.
        lims: Limits of the x axis.
        leaf_rotation: Rotation of the labels.
    """

    # Compute distance between one heatmap and the remaining, for all heatmaps
    # Distance matrix will have 0s on the diagonal and wil be symetric
    distances = [] 
    for i in range(len(heatmaps)-1):
        for j in range(i+1, len(heatmaps)):
            distances.append(distance_gray(heatmaps[i], heatmaps[j], 
                                           method=distance_method))
    
    # Perform hierarchical/agglomerative clustering
    links = linkage(distances, linkage_method)

    # Plot dendogram
    plt.rc('font', size=font_size)
    plt.figure(figsize=(width, height))
    plt.title(title, fontdict={'fontsize':title_size})
    dendrogram(links, labels=labels, orientation='right', 
               leaf_font_size=font_size, leaf_rotation=leaf_rotation)
    plt.tight_layout()
    if lims is not None:
        plt.xlim(lims[0], lims[1])
    plt.savefig(save_path)
    plt.clf()

    return

def plot_bar(labels, distances, title, save_path, sort=True, width=10, 
             height=15, title_size=20, font_size=16, reverse=True):
    """ Plot bar chart of distances.

    Args:
        labels: Labels of distances.
        distances: Distances to plot.
        title: Title of the plot.
        save_path: Path to save the plot.
        sort: Sort labels by distance.
        width: Width of the plot.
        height: Height of the plot.
        title_size: Size of the title.
        font_size: Size of the font.
        reverse: Reverse order of the bars.
    """
    # Sort if specified
    if sort:
        sorted_list = sorted(list(zip(labels, distances)), key=lambda t: t[1], reverse=reverse)
        labs, dists = [t[0] for t in sorted_list], [round(t[1], 2) for t in sorted_list]
    else:
        labs, dists = labels, distances

    # Plot
    plt.rc('font', size=font_size)
    plt.figure(figsize=(width, height))
    plt.title(title, fontdict={'fontsize':title_size})
    bars = plt.barh(width=dists, y=labs, color=(0.2, 0.4, 0.6, 1.0))
    plt.bar_label(bars)
    plt.savefig(save_path)
    plt.clf()

def heatmap_score(gt_bw, heatmap):
    """ Compute the score of a heatmap.

    Args:
        gt_bw: Binary ground truth image.
        heatmap: Heatmap to score.

    Returns:
        Score of the heatmap.
    """
    return np.sum(heatmap*gt_bw)/np.sum(heatmap)

def get_masks_distance(gt, th_heatmap, method='iou'):
    """ Compute distance between two binary images.

    Args:
        gt: Ground truth image.
        th_heatmap: Thresholded heatmap.
        method: Method to compute distance. It can be 'iou', 'f1_score', 
            'precision', 'recall' or 'accuracy'.

    Returns:
        Distance between the two images.
    """
    if method == 'iou':
        return iou(gt, th_heatmap)
    elif method == 'f1_score':
        return precision_recall_fscore_support(gt.flatten(), 
                                               th_heatmap.flatten(), average='binary')[2]
    elif method == 'precision':
        return precision_recall_fscore_support(gt.flatten(), 
                                               th_heatmap.flatten(), average='binary')[0]
    elif method == 'recall':
        return precision_recall_fscore_support(gt.flatten(), 
                                               th_heatmap.flatten(), average='binary')[1]
    elif method == 'accuracy':
        return accuracy(gt, th_heatmap)