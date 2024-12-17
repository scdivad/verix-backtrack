import pickle
import numpy as np
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.losses import CategoricalCrossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
import tf2onnx

import time
from VeriX import *
from VeriXBacktrack import *

"""
Load and process GTSRB data.
"""
gtsrb_path = 'models/gtsrb.pickle'
with open(gtsrb_path, 'rb') as handle:
    gtsrb = pickle.load(handle)
x_train, y_train = gtsrb['x_train'], gtsrb['y_train']
print(f"Original training data shape: {x_train.shape}, {y_train.shape}")  # (17550, 32, 32, 3), (17550,)
x_test, y_test = gtsrb['x_test'], gtsrb['y_test']
x_valid, y_valid = gtsrb['x_valid'], gtsrb['y_valid']

# Normalize images
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_valid = x_valid.astype('float32') / 255.0

# Downsample images to nxnx3 using TensorFlow
n = 10
x_train_resized = tf.image.resize(x_train, [n, n]).numpy()
x_test_resized = tf.image.resize(x_test, [n, n]).numpy()
x_valid_resized = tf.image.resize(x_valid, [n, n]).numpy()

gtsrb_labels = ['50 mph', '30 mph', 'yield', 'priority road',
                'keep right', 'no passing for large vechicles', '70 mph', '80 mph',
                'road work', 'no passing']

print(f"Resized training data shape: {x_train_resized.shape}")  # Should be (17550, n, n, 3)

# Convert labels to categorical format
num_classes = 10
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
y_valid = to_categorical(y_valid, num_classes)

choice = np.random.randint(low=0, high=x_test.shape[0], size=(100,))
for idx in choice:
    verix_base = VeriX(dataset="GTSRB",
                image=x_test_resized[idx],
                name=f"gtsrb_small_{idx}",
                model_path=f"models/gtsrb-{n}x{n}-10x2.onnx")
    verix_base.traversal_order(traverse="heuristic")
    with open("lookahead.txt", "a+") as f:
        f.write(f"baseline: {verix_base.get_explanation(epsilon=0.01)}\n")

    verix = VeriX2(dataset="GTSRB",
                image=x_test_resized[idx],
                name=f"gtsrb_small_{idx}",
                model_path=f"models/gtsrb-{n}x{n}-10x2.onnx")
    verix.traversal_order(traverse="heuristic")
    len_sat, len_timeout = verix.get_explanation(epsilon=0.01)

exit()

"""
Define and train a new model on the downsampled data.
"""
model_name = f'gtsrb-{n}x{n}-10x2'
model = Sequential(name=model_name)
model.add(Flatten(input_shape=(n, n, 3)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(num_classes))  # Output layer without activation (logits)
model.summary()

model.compile(
    loss=CategoricalCrossentropy(from_logits=True),
    optimizer=Adam(learning_rate=0.001),
    metrics=['accuracy']
)

datagen = ImageDataGenerator()
model.fit(
    datagen.flow(x=x_train_resized, y=y_train, batch_size=64),
    steps_per_epoch=100,
    epochs=60,
    validation_data=(x_valid_resized, y_valid),
    shuffle=True
)

# Evaluate the model on the test set
score = model.evaluate(x_test_resized, y_test, verbose=0)
print(f"Test loss: {score[0]}")
print(f"Test accuracy: {score[1]}")

# Save the model in ONNX format
model_proto, _ = tf2onnx.convert.from_keras(
    model, output_path='models/' + model_name + '.onnx'
)
