# Imports
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor
import tensorflow as tf

# Initialization
base_options = core.BaseOptions(file_name="./efficientnet_lite1_int8_2.tflite")
classification_options = processor.ClassificationOptions(max_results=2)
options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
classifier = vision.ImageClassifier.create_from_options(options)

print(classifier)

# Alternatively, you can create an image classifier in the following manner:
# classifier = vision.ImageClassifier.create_from_file(model_path)

# Run inference
image = vision.TensorImage.create_from_file("cat.jpeg")
classification_result = classifier.classify(image)

print(classification_result)