import matplotlib.pylab as plt
import tensorflow as tf
import numpy as np
from tflite_runtime.interpreter import Interpreter 

# Load MNIST dataset
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images.astype(np.float32) / 255.0
test_images = test_images.astype(np.float32) / 255.0

train_images = np.expand_dims(train_images, -1)
test_images = np.expand_dims(test_images, -1)


def classify(interpreter, x_test):
    test_image = x_test
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details["quantization"]
        test_image = x_test / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details["index"])[0]

    return output

model_path = './models/mnist_cnvnet_without_softmax.tflite'
interpreter = Interpreter(model_path)
interpreter.allocate_tensors()
print("Model Loaded Successfully.", end="\n\n")

for i in range(len(test_images)):
    print("correct label: ", test_labels[i])
    preds = classify(interpreter,test_images[i,::])
    print(preds)
    print("predicted label: ", np.argmax(preds))
    input("?")
