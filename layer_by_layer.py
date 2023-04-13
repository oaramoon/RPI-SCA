from oscilloscope import Oscilloscope
from trigger import Trigger
import numpy as np
import torchvision as tv
import torch
import torch.nn as nn
import numpy as np
import time


# Instantiate the scope 
scope = Oscilloscope()
scope.single()

# Instantiate the trigger
trigger = Trigger()

#### List available GPUs
available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
print(available_gpus)
#### Search for GPU
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


## Creating a random input to be fed to the model
sample_input = np.random.rand(16, 3, 224, 224).astype(np.float32)
sample_input = torch.Tensor(sample_input)


### Instantiating the model ###
output_size = 10
mother_model = tv.models.AlexNet()
mother_model.classifier[-1] = nn.Linear(4096, output_size)
mother_model.classifier.append(nn.Softmax(dim=1))

######################################################
########### Collecting EM traces #####################
######################################################

data_transfer_time = 80 #seconds
max_trigger_calls = 4
delay_between_queries = 3
num_removed_layers = 0
### Collecting the EM trace of the intact mother model
mother_model.to(device)
print("model's summary: ", mother_model.eval())

trigger_calls = 1
while True:
    d_inputs = sample_input.to(device)
    trigger.set()
    mother_model.forward(d_inputs)
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    if trigger_calls == max_trigger_calls:
        print("\nCollecting...")
        scope.save_waveform(name=str(num_removed_layers), wait_time=data_transfer_time)
        scope.single()
        time.sleep(delay_between_queries)
        break

    scope.single()
    time.sleep(delay_between_queries)
    trigger_calls += 1


#### Removing layers one by one and collecting the EM trace of modified model ### 
while len(list(mother_model.children())) != 0:

    child = list(mother_model.children())[-1]
    if type(child).__name__ == "Sequential":
        for k in range(len(child)-1,-1,-1):
            if k != 0:
                child[k] = nn.Identity()
            else:
                mother_model = nn.Sequential(*(nn.ModuleList(mother_model.children()))[:-1])
            
            mother_model.to(device)
            print("model's summary: ", mother_model.eval())
            
            trigger_calls = 1
            while True:
                d_inputs = sample_input.to(device)
                trigger.set()
                mother_model.forward(d_inputs)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if trigger_calls == max_trigger_calls:
                    print("\nCollecting...")
                    num_removed_layers += 1
                    scope.save_waveform(name=str(num_removed_layers), wait_time=data_transfer_time)
                    scope.single()
                    time.sleep(delay_between_queries)
                    break

                scope.single()
                time.sleep(delay_between_queries)
                trigger_calls += 1              
    else:
        
        mother_model = nn.Sequential(*(nn.ModuleList(mother_model.children()))[:-1])

        print("model summary: ", mother_model.eval())
        trigger_calls = 1
        try:
            while True:
                d_inputs = sample_input.to(device)
                trigger.set()
                mother_model.forward(d_inputs)
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
                if trigger_calls == max_trigger_calls:
                    print("\nCollecting...")
                    num_removed_layers += 1
                    scope.save_waveform(name=str(num_removed_layers), wait_time=data_transfer_time)
                    scope.single()
                    time.sleep(delay_between_queries)
                    break

                scope.single()
                time.sleep(delay_between_queries)
                trigger_calls += 1    
        except:
            pass



