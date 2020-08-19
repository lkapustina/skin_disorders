import numpy as np
import os
import cv2
import time
import glob
import warnings

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, log_loss, precision_score, recall_score, f1_score, classification_report

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import (Flatten, Dense, Conv2D, ZeroPadding2D, Dropout,
                          MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D)

from keras.optimizers import SGD
from keras.callbacks import EarlyStopping
from keras.utils import np_utils
from keras.layers import Input
from keras.applications.imagenet_utils import _obtain_input_shape

warnings.filterwarnings('ignore')

######################################################################################################
######################################################################################################


def read_and_resize_img(path_to_file, resize=None):
    img = cv2.imread(path_to_file)
    if resize:
        img = cv2.resize(img, (resize, resize), cv2.INTER_LINEAR)
    return img


def resize_img(img, resize):
    return cv2.resize(img, (resize, resize), cv2.INTER_LINEAR)


def load_train(size, folders):
    X_train = []
    X_train_id = []
    y_train = []
    start_time = time.time()

    print('Read train images')
    for fld in folders:
        index = folders.index(fld)
        path = os.path.join('data', fld, '*.jpg')
        files = glob.glob(path)
        print(f'Load folder "{fld}" with {len(files)} files (Index: {index})')
        for fl in files:
            flbase = os.path.basename(fl)
            img = read_and_resize_img(fl, size)
            X_train.append(img)
            X_train_id.append(flbase)
            y_train.append(index)

    print('Read train data time: {} seconds'.format(round(time.time() - start_time, 2)))
    return X_train, y_train, X_train_id


def read_and_normalize_train_data(size, folders):
    train_data, train_target, train_id = load_train(size, folders)

    print('Convert to numpy...')
    train_data = np.array(train_data, dtype=np.uint8)
    train_target = np.array(train_target, dtype=np.uint8)

    # if needed
    # print('Reshape...')
    # train_data = train_data.transpose((0, 3, 2, 1))

    print('Convert to float...')
    train_data = train_data.astype('float32')
    train_data = train_data / 255
    train_target = np_utils.to_categorical(train_target, len(folders))

    print('Train shape:', train_data.shape)
    print(train_data.shape[0], 'train samples')
    return train_data, train_target, train_id


def proba_to_classes(y_proba):
    if len(y_proba[0]) == 2:
        return np.array([list(map(round, item)) for item in y_proba])
    else:
        y_pred = []
        for lst in y_proba:
            max_lst = max(lst)
            for i, j in enumerate(lst):
                if j == max_lst:
                    max_ind = i
                    break
            lst1 = [0] * len(lst)
            lst1[max_ind] = 1
            y_pred.append(lst1)
        return np.array(y_pred)


def vgg16_example(classes, input_shape=None, include_top=True, pooling=None):
    """
    Model based architecture
    """

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=224,
                                      min_size=48,
                                      data_format=K.image_data_format(),
                                      require_flatten=include_top)

    img_input = Input(shape=input_shape)

    # Block 1
    x = Conv2D(4, (3, 3), activation='relu', padding='same', name='block1_conv1')(img_input)
    x = Conv2D(4, (3, 3), activation='relu', padding='same', name='block1_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

    # Block 2
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block2_conv1')(x)
    x = Conv2D(8, (3, 3), activation='relu', padding='same', name='block2_conv2')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

    # Block 3
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block3_conv1')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block3_conv2')(x)
    x = Conv2D(16, (3, 3), activation='relu', padding='same', name='block3_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

    # Block 4
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block4_conv1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block4_conv2')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block4_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

    # Block 5
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block5_conv1')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block5_conv2')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same', name='block5_conv3')(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

    if include_top:
        # Classification block
        x = Flatten(name='flatten')(x)
        x = Dense(128, activation='relu', name='fc1')(x)
        x = Dense(128, activation='relu', name='fc2')(x)
        x = Dense(classes, activation='softmax', name='predictions')(x)
    else:
        if pooling == 'avg':
            x = GlobalAveragePooling2D()(x)
        elif pooling == 'max':
            x = GlobalMaxPooling2D()(x)

    # Create model
    model = Model(img_input, x, name='custom_vgg16')
    return model


def small_VGG16(input_shape, n_classes):
    model = Sequential()

    model.add(Conv2D(4, (3, 3), activation='relu', input_shape=input_shape))
    model.add(Conv2D(4, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(Conv2D(8, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(Conv2D(16, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    return model


def deep_VGG16(input_shape, n_classes):
    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=input_shape))
    model.add(Conv2D(4, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(4, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(8, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(8, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(16, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(16, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(16, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Conv2D(32, 3, 3, activation='relu'))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(n_classes, activation='softmax'))

    return model


def main(deep=False):

    DATA_DIR = os.path.join(os.getcwd(), 'data')

    image_folders = os.listdir(DATA_DIR)
    image_folders.remove('imageUrls.p')
    image_folders.sort()

    images = os.listdir(os.path.join(DATA_DIR, image_folders[0]))
    images.sort()

    train_data, train_target, train_id = read_and_normalize_train_data(64, image_folders)

    num_classes = len(image_folders)

    X_train, X_test, y_train, y_test = train_test_split(train_data,
                                                        train_target,
                                                        test_size=0.25,
                                                        random_state=256)
    print('Training data', X_train.shape)

    print('Creating model...')
    if deep:
        model = deep_VGG16(X_train[0].shape, num_classes)
        model_name = 'deep_vgg16'
    else:
        model = small_VGG16(X_train[0].shape, num_classes)
        model_name = 'simple_vgg16'

    # model.compile(optimizer='rmsprop',
    #               loss='categorical_crossentropy',
    #               metrics=['accuracy'])

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd,
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=10),
    ]

    model.fit(X_train, y_train,
              batch_size=128,
              epochs=100,
              shuffle=True,
              verbose=1,
              validation_data=(X_test, y_test),
              callbacks=callbacks)

    y_proba = model.predict(X_test.astype('float32'),
                            batch_size=128)
    y_pred = proba_to_classes(y_proba)

    metrics = {
        'loglos': log_loss(y_test, y_proba),
        'accuracy': accuracy_score(y_test, y_pred),
        'precission': precision_score(y_test, y_pred, average='weighted'),
        'recall': recall_score(y_test, y_pred, average='weighted'),
        'f1_score': f1_score(y_test, y_pred, average='weighted')
    }

    print('***** Metrics *****\n')
    for k, v in metrics.items():
        print(f"{k}:\t {v:.4}".expandtabs(12))

    print(classification_report(y_test, y_pred, target_names=image_folders[:5]))

    model.save(f'models/{model_name}.h5')


if __name__ == '__main__':
    main()
