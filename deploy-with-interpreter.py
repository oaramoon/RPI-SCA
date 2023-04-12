from tflite_runtime.interpreter import Interpreter 
from PIL import Image
import numpy as np
import time
import matplotlib.pyplot


def set_input_tensor(interpreter, image):

    tensor_index = interpreter.get_input_details()[0]['index']
    print("Index of the input tensor: ", tensor_index, end="\n\n")

    # Return the input tensor based on its index.
    input_tensor = interpreter.tensor(tensor_index)()[0]
    print(input_tensor.shape)
    # Assigning the image to the input tensor.
    input_tensor[:, :] = image

def classify_image(interpreter, image):

  set_input_tensor(interpreter, image)

  # Call the invoke() method from inside a function to avoid this RuntimeError: reference to internal data in the interpreter in the form of a numpy array or slice.
  interpreter.invoke()

  output_details = interpreter.get_output_details()[0]
  print("\nDetails about the input tensors:\n   ", output_details, end="\n\n")

  scores = interpreter.get_tensor(output_details['index'])[0]
  print("Predicted class label score      =", np.max(np.unique(scores)))

  # Dequantize the scores.
  scale, zero_point = output_details['quantization']
  scores_dequantized = scale * (scores - zero_point)

  dequantized_max_score = np.max(np.unique(scores_dequantized))
  print("Predicted class label probability=", dequantized_max_score, end="\n\n")

  max_score_index = np.where(scores_dequantized == np.max(np.unique(scores_dequantized)))[0][0]
  print("Predicted class label ID=", max_score_index)

  return max_score_index, dequantized_max_score


data_folder = "mobilenet_v1_1.0_224_quant_and_labels/"

model_path = data_folder + "mobilenet_v1_1.0_224_quant.tflite"
label_path = data_folder + "labels_mobilenet_quant_v1_224.txt"

interpreter = Interpreter(model_path)
print("Model Loaded Successfully.", end="\n\n")

interpreter.allocate_tensors()
_, height, width, _ = interpreter.get_input_details()[0]['shape']
print("Input tensor size: (", width, ",", height, ")")

# Load an image to be classified.
image = Image.open("cat.jpeg").convert('RGB')
print("Original image size:", image.size)

image = image.resize((width, height))
print("New image size:", image.size, end="\n\n")

# Classify the image.
label_id, prob = classify_image(interpreter, image)

# Read class labels.
with open(label_path, 'r') as f:
  labels = [line.strip() for i, line in enumerate(f.readlines())]

# Return the classification label of the image.
classification_label = labels[label_id]
print("Image Label:", classification_label, "\nAccuracy   :", np.round(prob*100, 2), "%.")