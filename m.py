import numpy as np
import argparse
from tflite_runtime.interpreter import Interpreter 
#from trigger import Trigger
import pickle
import time



def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', dest='model', action="store" , type=str)
    args = parser.parse_args()
    
    with open("logit_dataset.pkl", 'rb') as f:
        logits_dataset = pickle.load(f)

    #### Instantiate the trigger
    #trigger = Trigger()
    
    model_path = args.model
    print("Model Path : ", model_path)
    interpreter = Interpreter(model_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    
    for i,(logit_sample,label) in enumerate(logits_dataset):
    
        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.uint8:
            input_scale, input_zero_point = input_details["quantization"]
            logit_sample = logit_sample / input_scale + input_zero_point
        
        logit_sample = np.expand_dims(logit_sample, axis=0).astype(input_details["dtype"])    
        interpreter.set_tensor(input_details["index"], logit_sample)
        collection_name = 'softmax-10-class-'+str(label)+'-'+str(i)
        
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        input(output)
    

if __name__ == '__main__':
    main()
