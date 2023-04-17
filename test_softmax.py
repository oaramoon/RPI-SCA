import numpy as np
import argparse
from tflite_runtime.interpreter import Interpreter 
from trigger import Trigger
import pickle
import time



def trigger_then_classify(interpreter, x_test, trigger):
    test_image = x_test
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    # Check if the input type is quantized, then rescale input data to uint8
    if input_details['dtype'] == np.uint8:
        input_scale, input_zero_point = input_details["quantization"]
        test_image = x_test / input_scale + input_zero_point

    test_image = np.expand_dims(test_image, axis=0).astype(input_details["dtype"])
    interpreter.set_tensor(input_details["index"], test_image)
    time.sleep(0.25)
    trigger.set()
    interpreter.invoke()
    trigger.clear()


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', action="store" , type=str)
    args = parser.parse_args()
    
    #### Instantiate the trigger
    trigger = Trigger()
    
    model_path = args.model
    print("Model Path : ", model_path)
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    print("Model Loaded Successfully.", end="\n\n")
    num_classes = int(args.model.split("/")[-1].split('.')[0].split('_')[1])
    print("Number of classes : ", num_classes)
    while True:
        d = float(input("delay between queries?\n>"))
        while True:
            trigger_then_classify(interpreter=interpreter,x_test=np.random.random((1,num_classes)),trigger=trigger)
            time.sleep(d)
    

if __name__ == '__main__':
    main()
