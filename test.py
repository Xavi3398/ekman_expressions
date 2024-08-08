from numpy import *
import os
import numpy as np
import json
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import shutil

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K


label_names = ['Anger', 'Disgust', 'Fear', 'Happiness', 'Sadness', 'Surprise']


def test_dir(test_path, save_path, model, img_size):
    """ Test a model with the specified parameters.

    Args:
        test_path: str. Path to the test dataset.
        save_path: str. Path to save the test predictions.
        model: Model. Trained model.
        img_size: int. Image size.
    """

    datagen = ImageDataGenerator(rescale=1./255)     

    # Test set
    generator = datagen.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        class_mode='categorical',
        shuffle=False)  

    # Predict all test set
    K.set_image_data_format('channels_last')
    predictions = model.predict(generator)

    # Save predictions to json
    with open(save_path, 'w') as outfile:
        json.dump(predictions.tolist(), outfile)


def plot_confusion_matrix(test_path, test_json, img_size, display_labels,
                          cm_path):
    """ Plot and save the confusion matrix.

    Args:
        test_path: str. Path to the test dataset.
        test_json: str. Path to the test predictions.
        img_size: int. Image size.
        display_labels: list of str. Labels to display.
        cm_path: str. Path to save the confusion matrix.
    """

    # Get ground truth and predictions
    y_gt, y_pred = load_y(test_path, test_json, img_size)

    # Compute Confusion Matrix
    cm = confusion_matrix(y_gt, y_pred)

    # Display and save Confusion Matrix
    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm, 
        display_labels=display_labels)
    disp.plot(cmap='Blues')
    plt.savefig(cm_path)
    plt.close()
    plt.clf()


def save_classification_report(test_path, test_json, img_size, out_path):
    """ Save the classification report to a json file.

    Args:
        test_path: str. Path to the test dataset.
        test_json: str. Path to the test predictions.
        img_size: int. Image size.
        out_path: str. Path to save the classification report.
    """

    # Get ground truth and predictions
    y_gt, y_pred = load_y(test_path, test_json, img_size)

    # Get classification report
    cr = classification_report(y_gt, y_pred, output_dict=True)

    # Save classification report to json
    with open(out_path, 'w') as outfile:
        json.dump(cr, outfile)


def load_y(test_path, test_json, img_size):
    """ Load ground truth and predictions.

    Args:
        test_path: str. Path to the test dataset.
        test_json: str. Path to the test predictions.
        img_size: int. Image size.

    Returns:
        y_gt: np.array. Ground truth.
        y_pred: np.array. Predictions.
    """

    # Test set for ground truth
    datagen = ImageDataGenerator(rescale=1./255)   
    generator = datagen.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        class_mode='categorical',
        shuffle=False)  
    y_gt = generator.classes

    # Load test predictions and choose predicted class
    with open(test_json, 'r') as f:
        y_pred = np.array(json.load(f))
        y_pred = np.argmax(y_pred, axis=1)
    
    return y_gt, y_pred

def save_test_imgs(test_path, test_json, img_size, label_names, save_path, save_TF=True):
    """ Save test images to a directory.

    Args:
        test_path: str. Path to the test dataset.
        test_json: str. Path to the test predictions.
        img_size: int. Image size.
        label_names: list of str. Labels.
        save_path: str. Path to save the test images.
        save_TF: bool. Save images with True/False prefix.
    """

    # Test set for files
    datagen = ImageDataGenerator(rescale=1./255)   
    generator = datagen.flow_from_directory(
        test_path,
        target_size=(img_size, img_size),
        class_mode='categorical',
        shuffle=False)  
    y_files = generator.filepaths
    y_gt = generator.classes

    # Load test predictions and choose predicted class
    with open(test_json, 'r') as f:
        y_pred = np.array(json.load(f))
        y_pred = np.argmax(y_pred, axis=1)
    
    # Create a subdirectory for each class
    for label in label_names:
        if not os.path.exists(os.path.join(save_path, label)):
            os.mkdir(os.path.join(save_path, label))
    
    # Save images
    for f, y_p, y_t in zip(y_files, y_pred, y_gt):
        if save_TF:
            f_name = ('T' if y_p == y_t else 'F') + '_' + os.path.basename(f)
        else:
            f_name = os.path.basename(f)
        shutil.copyfile(f, os.path.join(save_path, label_names[y_p], f_name))
                