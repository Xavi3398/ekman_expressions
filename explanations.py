from PIL import Image
import numpy as np
import os
import cv2
import shutil
import json

from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.segmentation import felzenszwalb, slic, quickshift

from ekman_expressions.lime_image import LimeImageExplainer
from ekman_expressions.utils import painter, segments2colors


def save_positives_per_class(test_path, test_json, label_names, save_path,
                                 n_positives=100):
    """ Save n positives per class in a directory.

    Args:
        test_path: str. Path to the test dataset.
        test_json: str. Path to the test predictions.
        label_names: list of str. Labels.
        save_path: str. Path to save the positives.
        n_positives: int. Number of positives per class.
    """

    # Load test predictions and choose predicted class
    with open(test_json, 'r') as f:
        y_pred = np.array(json.load(f))
        y_pred = np.argmax(y_pred, axis=1)
    
    # Load test paths
    datagen = ImageDataGenerator(rescale=1./255)   
    generator = datagen.flow_from_directory(
        test_path,
        class_mode='categorical',
        shuffle=False)
    img_paths = generator.filepaths
    
    # Get list of positives per class
    class_positives = [[] for _ in range(len(label_names))]
    for img, pred in zip(img_paths, y_pred):
        class_positives[pred].append(img)

    for i in range(len(label_names)):

        # Create the class directory
        class_dir = os.path.join(save_path, str(i))
        if not os.path.exists(class_dir):
            os.mkdir(class_dir)
        save_dir = os.path.join(class_dir, 'positives')
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        # Randomly choose n postives (or maximum of positives if it is smaller)
        # per class
        n_choices = min(n_positives, len(class_positives[i]))
        imgs = np.random.choice(class_positives[i], n_choices, replace=False)

        # Copy the chosen images
        for img_path in imgs:
            shutil.copy(img_path, os.path.join(save_dir, os.path.basename(img_path)))

def apply_lime(img_path, img_size, model, exp_img_path, exp_mask_path, exp_seg_path,
             segmentation_fn=None, hide_color=0, num_samples=1000, label=None,
             th=None, top_k=None, min_accum=None, improve_background=False, pos_only=False, neg_only=False,
             hist_stretch=True, invert=True):
    """ Apply LIME explanation to an image.

    Args:
        img_path: str. Path to the image.
        img_size: int. Image size.
        model: Model. Model to explain.
        exp_img_path: str. Path to save the explanation image.
        exp_mask_path: str. Path to save the explanation mask.
        exp_seg_path: str. Path to save the explanation segmentation.
        segmentation_fn: function. Segmentation function.
        hide_color: int. Color used to mask out regions.
        num_samples: int. Number of samples.
        label: int. Label to explain.
        th: float. Threshold to use to delete lesser important regions.
        top_k: int. Leave only top k most important regions.
        min_accum: int. Leave only regions summing up to a minimum accumulation importance.
        improve_background: bool. Improve background.
        pos_only: bool. Leave only positive importance.
        neg_only: bool. Leave only negative importance.
        hist_stretch: bool. Use histogram stretching to improve the explanation.
        invert: bool. Invert importance (lower values are more important).
    """
    
    # Set Keras config
    K.set_image_data_format('channels_last')

    # Load, resize and convert image
    img = Image.open(img_path)
    img = img.resize((img_size, img_size), resample=0)
    img = img.convert("RGB")
    img = np.array(img) / 255

    # Segment
    if segmentation_fn is None:
        segments = slic(img, n_segments=30, compactness=20.0, start_label=0)
    elif segmentation_fn == 'quickshift':
        segments = quickshift(img, kernel_size=4, max_dist=200, ratio=0.2)
    elif segmentation_fn == 'slic':
        segments = slic(img, n_segments=30, compactness=20.0, start_label=0)
    else:
        segments = segmentation_fn(img)

    # Compute LIME explanation
    explainer = LimeImageExplainer()
    explanation = explainer.explain_instance(
        img, 
        model,
        segments=segments,
        top_labels=1,
        labels=[label],
        hide_color=hide_color, 
        num_samples=num_samples,
        progress_bar=False)
    
    # Stop if we are running explanation on the wrong label
    if label is not None and label != explanation.top_labels[0]:
        print('Error: label=' + str(label) + ' with conf=' + str(explanation.gt[label]) + ' but max_conf_label=' + str(explanation.top_labels[0]) + ' with conf=' + str(explanation.gt[explanation.top_labels[0]]) + ' on ' + img_path)
        # raise Exception('Error: label=' + str(label) + ' but max_conf_label=' + str(explanation.top_labels[0]) + ' on ' + img_path)

    score_map = explanation.get_score_map(
        explanation.top_labels[0] if label is None else label, 
        th, 
        top_k, 
        min_accum, 
        improve_background, 
        pos_only, 
        neg_only)
    score_map_rgb = explanation.get_score_map_rgb(
        score_map, 
        hist_stretch, 
        invert)

    # Save LIME explanation image and mask
    cv2.imwrite(os.path.join(exp_seg_path, os.path.basename(img_path)[:-4] + '_seg.png'), segments2colors(segments, img))
    cv2.imwrite(os.path.join(exp_mask_path, os.path.basename(img_path)[:-4] + '_mask.png'), (score_map*255).astype('uint8'))
    cv2.imwrite(os.path.join(exp_img_path, os.path.basename(img_path)[:-4] + '_exp.png'), cv2.cvtColor(painter((img*255).astype('uint8'), score_map_rgb), cv2.COLOR_RGB2BGR))
