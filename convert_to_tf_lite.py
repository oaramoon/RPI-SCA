import tensorflow as tf
from tensorflow import keras
import numpy as np

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)

base_model = keras.models.load_model('mnist_cnvnet.hdf5')

converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
tflite_model = converter.convert()
with open('./models/mnist_cnvnet_with_softmax.tflite', 'wb') as f:
    f.write(tflite_model)
    
   
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(1000):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()
with open('./models/mnist_cnvnet_with_softmax_uint8.tflite', 'wb') as f:
    f.write(tflite_model_quant)
    


base_model.pop()
input(base_model.summary())

converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
tflite_model = converter.convert()
with open('./models/mnist_cnvnet_without_softmax.tflite', 'wb') as f:
    f.write(tflite_model)
    
   
def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(train_images).batch(1).take(1000):
    yield [input_value]

converter = tf.lite.TFLiteConverter.from_keras_model(base_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_data_gen
# Ensure that if any ops can't be quantized, the converter throws an error
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
# Set the input and output tensors to uint8 (APIs added in r2.3)
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8

tflite_model_quant = converter.convert()
with open('./models/mnist_cnvnet_without_softmax_uint8.tflite', 'wb') as f:
    f.write(tflite_model_quant)