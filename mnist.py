from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical
from keras.datasets import mnist
from keras.losses import CategoricalCrossentropy
from keras.optimizers import Adam
import tf2onnx
from VeriX import *
from VeriXBacktrack import *

"""
download and process MNIST data.
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

"""
show a simple example usage of VeriX. 
"""
choice = np.random.randint(low=0, high=x_test.shape[0], size=(100,))
for idx in choice:
    verix_base = VeriX(dataset="MNIST",
                    image=x_test[idx],
                    name=f"mnist_{idx}",
                    model_path="models/mnist-10x2.onnx",
                    epsilon=0.05)
    verix_base.traversal_order(traverse="heuristic")
    with open("lookahead_mnist.txt", "a+") as f:
        f.write(f"baseline: {verix_base.get_explanation(epsilon=0.05)}\n")

    verix = VeriX2(dataset="MNIST",
                image=x_test[idx],
                name=f"mnist_{idx}",
                model_path="models/mnist-10x2.onnx")
    verix.traversal_order(traverse="heuristic")

    len_sat, len_timeout = verix.get_explanation(epsilon=0.05)
exit()

"""
or you can train your own MNIST model.
Note: to obtain sound and complete explanations, train the model from logits directly.
"""
from tensorflow.keras.models import load_model

# Load the saved model
model_name = 'mnist-10x2.onnx'
loaded_model = load_model('models/' + model_name)

# Evaluate the loaded model
score = loaded_model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
exit()



model_name = 'mnist-10x2'
model = Sequential(name=model_name)
model.add(Flatten(input_shape=(28, 28, 1)))
model.add(Dense(10, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(10))
model.summary()
model.compile(loss=CategoricalCrossentropy(from_logits=True),
              optimizer=Adam(learning_rate=0.001),
              metrics=['accuracy'])
model.fit(x_train, y_train,
          batch_size=128,
          epochs=20,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
model.save('models/' + model_name + '.h5')
# model_proto, _ = tf2onnx.convert.from_keras(model, output_path='models/' + model_name + '.onnx')




