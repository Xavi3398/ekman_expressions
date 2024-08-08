from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
import tensorflow as tf
import keras.backend as tfback

from ekman_expressions.nets import getNetByName

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    Returns:
        A list of available GPU devices.
    """
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]


def train(model_name, dataset_path, save_path, n_classes=6, dataset_path_val=None,
          batch_size_train=128, batch_size_val=32, validation_split=0.25,
          rescale=1./255, verbose=True, epochs=45, save_weights=False, augment=False):
    """ Train a model with the specified parameters.

    Args:
        model_name: str. Name of the model to train.
        dataset_path: str. Path to the training dataset.
        save_path: str. Path to save the trained model.
        n_classes: int. Number of classes in the dataset.
        dataset_path_val: str. Path to the validation dataset.
        batch_size_train: int. Batch size for training.
        batch_size_val: int. Batch size for validation.
        validation_split: float. Fraction of images reserved for validation.
        rescale: float. Rescale factor.
        verbose: bool. Print training info.
        epochs: int. Number of training epochs.
        save_weights: bool. Save model weights.
        augment: bool. Apply data augmentation.
    """

    # Configure Keras/TensorFlow settings
    K.set_image_data_format('channels_last')
    tfback._get_available_gpus = _get_available_gpus

    # Model and image size selection 
    model, image_size = getNetByName(model_name, n_classes, augment)

    # Image generator settings
    datagen = ImageDataGenerator(
        validation_split=validation_split if dataset_path_val is None else 0.0,
        rescale=rescale)     

    # Training set
    train_generator = datagen.flow_from_directory(
        dataset_path,  
        subset='training' if dataset_path_val is None else None,
        target_size=(image_size, image_size),  
        batch_size=batch_size_train, 
        class_mode='categorical')  

    # Validation set
    validation_generator = datagen.flow_from_directory(
        dataset_path if dataset_path_val is None else dataset_path_val,
        subset='validation' if dataset_path_val is None else None,
        target_size=(image_size, image_size),
        batch_size=batch_size_val, 
        class_mode='categorical')

    # Steps per epoch for training and validation
    training_steps = train_generator.n//batch_size_train if train_generator.n > batch_size_train else 1
    validation_steps = validation_generator.n//batch_size_val if validation_generator.n > batch_size_val else 1

    # Print info before training
    if verbose:
        print('========================================')
        print('Net:', model_name)
        print('Dataset:', dataset_path)
        print('Training images:', train_generator.n)
        print('Validation images:', validation_generator.n)
        print('Steps per epoch (training):', training_steps)
        print('Steps per epoch (validation):', validation_steps)
        print('Model save path:', save_path + '_model.h5')
        print('Weights save path:', save_path + '_weights.h5')
        print('========================================')

    # Training
    model.fit(
        train_generator,
        steps_per_epoch= training_steps, 
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps= validation_steps,
        workers=3) 

    # Save model
    # model.save(save_path + '_model.h5')

    # # Save weights
    # if save_weights:
    model.save_weights(save_path + '_weights.h5')
