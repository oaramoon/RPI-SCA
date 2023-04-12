# Imports
from tflite_support.task import vision
from tflite_support.task import core
from tflite_support.task import processor
import time
import argparse
from trigger import Trigger


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', action="store" , type=str)

    args = parser.parse_args()

    #### Instantiate the trigger
    trigger = Trigger()

    model_path = args.model
    print("Model Path : ", model_path)
    
    # Initialization
    base_options = core.BaseOptions(file_name=model_path)
    classification_options = processor.ClassificationOptions(max_results=2)
    options = vision.ImageClassifierOptions(base_options=base_options, classification_options=classification_options)
    classifier = vision.ImageClassifier.create_from_options(options)

    image = vision.TensorImage.create_from_file("cat.jpeg")
    
    while True:
        d = float(input("delay between queires?\n>"))
        while True:
            trigger.set()
            start = time.time()
            preds = classifier.classify(image)
            end = time.time()
            print("inference time: ", end-start , "predictions: ", preds)
            trigger.clear()
            time.sleep(d)


if __name__ == '__main__':
    main()
