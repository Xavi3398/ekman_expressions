from keras.models import Sequential
from keras.layers import *
from keras import applications as kapps
from keras.optimizers import RMSprop, Adam


def getNetByName(model_name, n_classes=6, augment=False):
    """ Get a model by name.

    Args:
        model_name: str. Name of the model.
        n_classes: int. Number of classes.
        augment: bool. Apply data augmentation.

    Returns:
        model: Keras model.
        image_size: int. Image size.
    """

    if model_name == 'SilNet':
        model = SilNet(n_classes, augment)
        image_size = 150
    elif  model_name == 'WeiNet':
        model = WeiNet(n_classes, augment)
        image_size = 64
    elif  model_name == 'AlexNet':
        model = AlexNet(n_classes, augment)
        image_size = 224
    elif  model_name == 'SongNet':
        model = SongNet(n_classes, augment)
        image_size = 224
    elif  model_name == 'InceptionV3':
        model = InceptionV3(n_classes, augment)
        image_size = 224
    elif  model_name == 'VGG19':
        model = VGG19(n_classes, augment)
        image_size = 224
    elif  model_name == 'VGG16':
        model = VGG16(n_classes, augment)
        image_size = 224
    elif  model_name == 'ResNet50':
        model = ResNet50(n_classes, augment)
        image_size = 224
    elif  model_name == 'ResNet101V2':
        model = ResNet101V2(n_classes, augment)
        image_size = 224
    elif  model_name == 'Xception':
        model = Xception(n_classes, augment)
        image_size = 224
    elif  model_name == 'MobileNetV3Large':
        model = MobileNetV3Large(n_classes, augment)
        image_size = 224
    elif  model_name == 'EfficientNetV2B0':
        model = EfficientNetV2B0(n_classes, augment)
        image_size = 224
    elif  model_name == 'ConvNeXtTiny':
        model = ConvNeXtTiny(n_classes, augment)
        image_size = 224
    
    return model, image_size


def addDataAugmentationLayers(model):
    """ Add data augmentation layers to a model.

        Args:
            model: Keras model.
    """
    model.add(RandomFlip("horizontal"))
    model.add(RandomRotation(0.03, fill_mode='constant', fill_value=255.0))
    model.add(RandomTranslation(0.05, 0.05, fill_mode='constant', fill_value=255.0))
    model.add(RandomContrast(0.1))


def getDataAugmentationLayers():
    """ Get data augmentation layers.
        
        Returns:
            Sequential. Data augmentation layers.
    """
    return Sequential([
        RandomFlip("horizontal"),
        RandomRotation(0.03, fill_mode='constant', fill_value=255.0),
        RandomTranslation(0.05, 0.05, fill_mode='constant', fill_value=255.0),
        RandomContrast(0.1)
    ])


def AlexNet(n_classes=6, augment=False):
    """ AlexNet model.

    Args:
        n_classes: int. Number of classes.
        augment: bool. Apply data augmentation.

    Returns:
        model: Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    if augment:
         addDataAugmentationLayers(model)

    model.add(Conv2D(96, kernel_size = 11, strides= (4, 4), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2,2), padding='same'))

    model.add(Conv2D(256, kernel_size = 5, strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))

    model.add(Conv2D(384, kernel_size = 3, strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))

    model.add(Conv2D(384, kernel_size = 3, strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))

    model.add(Conv2D(256, kernel_size = 3, strides=(1, 1), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='same'))

    model.add(Flatten())
    model.add(Dense(4096, input_shape=(224*224*3,), activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.4))

    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])

    return model


def WeiNet(n_classes=6, augment=False):
    """ WeiNet model.

    Args:
        n_classes: int. Number of classes.
        augment: bool. Apply data augmentation.

    Returns:
        model: Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(64, 64, 3)))

    if augment:
         addDataAugmentationLayers(model)

    model.add(Conv2D(32, (7, 7), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    model.add(Conv2D(32, (7, 7), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    model.add(Conv2D(64, (7, 7), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])
    return model


def SongNet(n_classes=6, augment=False):
    """ SongNet model.

    Args:
        n_classes: int. Number of classes.
        augment: bool. Apply data augmentation.

    Returns:
        model: Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    if augment:
         addDataAugmentationLayers(model)

    model.add(Conv2D(64, (5, 5), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Conv2D(64, (5, 5), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))

    model.add(Conv2D(32, (3, 3), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Conv2D(32, (3, 3), padding='same', data_format='channels_last', activation='relu'))
    model.add(MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same'))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=RMSprop(lr=1e-4), metrics=['accuracy'])
    return model


def SilNet(n_classes=6, augment=False):
    """ SilNet model.

    Args:
        n_classes: int. Number of classes.
        augment: bool. Apply data augmentation.

    Returns:
        model: Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(150, 150, 3)))

    if augment:
         addDataAugmentationLayers(model)

    model.add(Conv2D(32, (11, 11), padding='same', data_format='channels_last', kernel_initializer='glorot_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same'))

    model.add(Conv2D(32, (7, 7), padding='same', data_format='channels_last', kernel_initializer='glorot_normal', activation='relu'))

    model.add(Conv2D(32, (7, 7), padding='same', data_format='channels_last', kernel_initializer='glorot_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    model.add(Conv2D(64, (5, 5), padding='same', data_format='channels_last', kernel_initializer='glorot_normal', activation='relu'))
    
    model.add(Conv2D(64, (5, 5), padding='same', data_format='channels_last', kernel_initializer='glorot_normal', activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same'))

    model.add(Flatten())
    model.add(Dense(512, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(512, kernel_initializer='glorot_normal', activation='relu'))
    model.add(Dense(n_classes, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4), metrics=['accuracy'])
    return model


def InceptionV3(n_classes=6, augment=False):
    """ InceptionV3 model.

    Args:
        n_classes: int. Number of classes.
        augment: bool. Apply data augmentation.

    Returns:
        model: Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    if augment:
         addDataAugmentationLayers(model)

    model.add(kapps.InceptionV3(weights='imagenet', include_top=False))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

    return model


def VGG19(n_classes=6, augment=False):
    """ VGG19 model.

    Args:
        n_classes: int. Number of classes.
        augment: bool. Apply data augmentation.

    Returns:
        model: Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    if augment:
         addDataAugmentationLayers(model)

    model.add(kapps.VGG19(weights='imagenet', include_top=False))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

    return model


def VGG16(n_classes=6, augment=False):
    """ VGG16 model.

    Args:
        n_classes: int. Number of classes.
        augment: bool. Apply data augmentation.

    Returns:
        model: Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    if augment:
         addDataAugmentationLayers(model)

    model.add(kapps.VGG16(weights='imagenet', include_top=False))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

    return model


def ResNet50(n_classes=6, augment=False):
    """ ResNet50 model.

    Args:
        n_classes: int. Number of classes.
        augment: bool. Apply data augmentation.

    Returns:
        model: Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    if augment:
         addDataAugmentationLayers(model)

    model.add(kapps.ResNet50(weights='imagenet', include_top=False))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

    return model


def ResNet101V2(n_classes=6, augment=False):
    """ ResNet101V2 model.

    Args:
        n_classes: int. Number of classes.
        augment: bool. Apply data augmentation.

    Returns:
        model: Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    if augment:
         addDataAugmentationLayers(model)

    model.add(kapps.ResNet101V2(weights='imagenet', include_top=False))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

    return model


def Xception(n_classes=6, augment=False):
    """ Xception model.

    Args:
        n_classes: int. Number of classes.
        augment: bool. Apply data augmentation.

    Returns:
        model: Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    if augment:
         addDataAugmentationLayers(model)

    model.add(kapps.Xception(weights='imagenet', include_top=False))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-4), metrics=['accuracy'])

    return model


def MobileNetV3Large(n_classes=6, augment=False):
    """ MobileNetV3Large model.

    Args:
        n_classes: int. Number of classes.
        augment: bool. Apply data augmentation.

    Returns:
        model: Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    if augment:
         addDataAugmentationLayers(model)
         
    model.add(kapps.MobileNetV3Large(weights='imagenet', include_top=False))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-6), metrics=['accuracy'])

    return model


def EfficientNetV2B0(n_classes=6, augment=False):
    """ EfficientNetV2B0 model.

    Args:
        n_classes: int. Number of classes.
        augment: bool. Apply data augmentation.

    Returns:
        model: Keras model.
    """
    model = Sequential()
    model.add(Input(shape=(224, 224, 3)))

    if augment:
         addDataAugmentationLayers(model)

    model.add(kapps.EfficientNetV2B0(weights='imagenet', include_top=False))
    model.add(GlobalAveragePooling2D())
    model.add(Dense(n_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=1e-6), metrics=['accuracy'])

    return model
