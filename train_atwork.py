import numpy as np
from keras_squeezenet import SqueezeNet
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.preprocessing import image
import glob
from keras.preprocessing import image as k_image
import numpy as np
from keras.layers import Activation, Conv2D, Dropout, Dense, Flatten
from keras.layers import AveragePooling2D, BatchNormalization, MaxPooling2D
from keras.layers import GlobalAveragePooling2D
from keras.models import Sequential
from keras.callbacks import ModelCheckpoint
from keras import optimizers
import matplotlib.pyplot as plt

def display_image(image_array, title):
    image_array = np.squeeze(image_array)
    plt.imshow(image_array)
    plt.title(str(title))
    plt.show()

def load_images(image_paths, target_size, grayscale=False):
    images = []
    for image_path in image_paths:
        image = k_image.load_img(image_path, grayscale, target_size)
        image_array = k_image.img_to_array(image)
        images.append(image_array)
    return np.asarray(images)

def to_categorical(data):
    arg_classes = np.unique(data).tolist()
    arg_classes.sort()
    num_classes = len(arg_classes)
    label_to_arg = dict(zip(arg_classes, list(range(num_classes))))
    num_samples = len(data)
    categorical_data = np.zeros(shape=(num_samples, num_classes))
    for sample_arg in range(num_samples):
        label = data[sample_arg]
        data_arg = label_to_arg[label]
        categorical_data[sample_arg, data_arg] = 1
    return categorical_data

def preprocess_images(images):
    return images/255.

# parameters
image_size = (227, 227, 3)
path = '/data/dev/robocup/nagoya_segmentation/F20_20_Axis_recognition/'
trained_models_path = 'classifier'
batch_size = 32
num_epochs = 1000

# spitting up code... sorry
train_f20_path = path + 'train/F20_20_G/'
train_axis_path = path + 'train/AXIS/'
validation_f20_path = path + 'validation/F20_20_G/'
validation_axis_path = path + 'validation/AXIS/'
train_f20_filenames = glob.glob(train_f20_path + '*.jpg')
train_axis_filenames = glob.glob(train_axis_path + '*.jpg')
validation_f20_filenames = glob.glob(validation_f20_path + '*.jpg')
validation_axis_filenames = glob.glob(validation_axis_path + '*.jpg')
train_f20_classes = np.repeat(0, len(train_f20_filenames))
train_axis_classes = np.repeat(1, len(train_axis_filenames))
train_classes = to_categorical(np.concatenate([train_f20_classes, train_axis_classes]))
validation_f20_classes = np.repeat(0, len(validation_f20_filenames))
validation_axis_classes = np.repeat(1, len(validation_axis_filenames))
validation_classes = to_categorical(np.concatenate([validation_f20_classes, validation_axis_classes]))
train_f20_images =  load_images(train_f20_filenames, image_size[:2])
train_axis_images =  load_images(train_axis_filenames, image_size[:2])
validation_f20_images =  load_images(validation_f20_filenames, image_size[:2])
validation_axis_images =  load_images(validation_axis_filenames, image_size[:2])
train_images = np.concatenate([train_f20_images, train_axis_images], axis=0)
train_images = preprocess_images(train_images)
validation_images = np.concatenate([validation_f20_images, validation_axis_images], axis=0)
validation_images = preprocess_images(validation_images)
train_data = (train_images, train_classes)
validation_data = (validation_images, validation_classes)

model = SqueezeNet(weights=None, input_shape=image_size, classes=2)
opt = optimizers.SGD(lr=0.001)
model.compile(loss = "categorical_crossentropy", optimizer = opt, metrics=['accuracy'])
#model.compile(optimizer='adam', loss='categorical_crossentropy',
#                                        metrics=['accuracy'])

model.summary()
model_names = trained_models_path + '.{epoch:02d}-{val_acc:.2f}.hdf5'
model_checkpoint = ModelCheckpoint(model_names,
                            'val_acc', verbose=1,
                                save_best_only=True)
callbacks = [model_checkpoint]

model.fit(train_images, train_classes, batch_size, num_epochs,
            verbose=1,
            callbacks=callbacks,
            validation_data=validation_data, shuffle=True)
