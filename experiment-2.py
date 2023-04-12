import numpy as np
import argparse
import tensorflow as tf
import numpy as np
from tflite_runtime.interpreter import Interpreter 
from trigger import Trigger
import pickle



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
    
    trigger.set()
    interpreter.invoke()
    trigger.clear()
    output = interpreter.get_tensor(output_details["index"])[0]

    return output


def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', action="store" , type=str)
    parser.add_argument('--num_collections', action="store", dest='num_collections',  type=int, default=3)
    
    args = parser.parse_args()
    
    with open("1_vs_8.pkl",'rb') as f:
        data = pickle.load(f)


    #### Instantiate the trigger
    trigger = Trigger()
    
    model_path = args.model
    print("Model Path : ", model_path)
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    print("Model Loaded Successfully.", end="\n\n")

    for i in range(args.num_collections):
        preds = trigger_then_classify(interpreter=interpreter,x_test=data['8'][i,::],trigger=trigger)
        print("Predictions: ", preds)
        print("Predicted Label: ", np.argmax(preds))
        collection_name = model_path.split("/")[-1].split('.')[0]+'-class-8-'+str(i)
        print("Collect the EM Trace ", collection_name)
        input("?")
        

    for i in range(args.num_collections):
        preds = trigger_then_classify(interpreter=interpreter,x_test=data['1'][i,::],trigger=trigger)
        print("Predictions: ", preds)
        print("Predicted Label: ", np.argmax(preds))
        collection_name = model_path.split("/")[-1].split('.')[0]+'-class-1-'+str(i)
        print("Collect the EM Trace ", collection_name)
        input("?")
    

if __name__ == '__main__':
    main()
