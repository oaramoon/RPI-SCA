from oscilloscope import Oscilloscope
from trigger_gpu import Trigger
import numpy as np
import torchvision
import torch
import torch.nn as nn
import torch.nn.functional as F  # Added this import
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.data import ConcatDataset
from tensorflow.keras.datasets import mnist
import argparse
import time

def main():
    # Instantiate the scope 
    scope = Oscilloscope()
    scope.single()

    # Instantiate the trigger
    trigger = Trigger()

    #### List available GPUs
    available_gpus = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    print("available gpus : ", available_gpus)
    #### allocating the GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    #### Instantiate the model
    # AlexNet for CIFAR10
    class AlexNet(nn.Module):
        def __init__(self, num_classes=10):
            super(AlexNet, self).__init__()
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=5),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(64, 192, kernel_size=5, padding=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Conv2d(192, 384, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(384, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2, stride=2),
            )
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

        def forward(self, x):
            x = self.features(x)
            x = x.view(x.size(0), -1)
            x = self.classifier(x)
            return F.log_softmax(x, dim=1)  # Apply log_softmax

    model = AlexNet()
    # Load the trained weights
    model.load_state_dict(torch.load('cifar10_alexnet.pth'))
    model.eval()  # Set the model to evaluation mode
    model.to(device)

    ### Collecting EM traces ###
    
    # Load CIFAR10 dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Load CIFAR10 datasets
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

    # Concatenate train and test datasets
    combined_dataset = ConcatDataset([train_dataset, test_dataset])

    # Create a DataLoader for the combined dataset
    combined_loader = torch.utils.data.DataLoader(combined_dataset, batch_size=1, shuffle=False)


    data_transfer_time = 16
    max_trigger_calls = 7
    delay_between_queries = 0.85
    cut = 30000
    # Loop through samples

    num_samples = 0

    for idx, (sample, target) in enumerate(list(combined_loader)[:cut]):
        
        if not target.item() in [1,8]:
            continue

        input_sample = sample.to(device)      
        
        trigger_calls = 1
        while True:
            trigger.set()
            output = model.forward(input_sample)
            
            if trigger_calls == max_trigger_calls:
                print("\nCollecting...")
                confidence, predicted_class = torch.exp(output).max(dim=1)
                collection_name = 'cifar10-class-'+str(target.item())+'-predicted-as-'+str(predicted_class.item())+\
                "-with-confidence-"+"{:.2f}".format(100*confidence.item())+"-"+str(idx)
                time.sleep(1.0)
                scope.save_waveform_D(name=collection_name, wait_time=data_transfer_time)
                time.sleep(1)
                scope.single()
                time.sleep(delay_between_queries)
                break

            scope.single()
            time.sleep(delay_between_queries)
            trigger_calls += 1


    # Loop through samples
    for idx, (sample, target) in enumerate(list(combined_loader)[cut:],start=cut):

        if not target.item() in [1,8]:
            continue

        
        input_sample = sample.to(device)      
        
        trigger_calls = 1
        while True:
            trigger.set()
            output = model.forward(input_sample)
            
            if trigger_calls == max_trigger_calls:
                print("\nCollecting...")
                confidence, predicted_class = torch.exp(output).max(dim=1)
                collection_name = 'cifar10-class-'+str(target.item())+'-predicted-as-'+str(predicted_class.item())+\
                "-with-confidence-"+"{:.2f}".format(100*confidence.item())+"-"+str(idx)
                time.sleep(1.0)
                scope.save_waveform_E(name=collection_name, wait_time=data_transfer_time)
                time.sleep(1)
                scope.single()
                time.sleep(delay_between_queries)
                break

            scope.single()
            time.sleep(delay_between_queries)
            trigger_calls += 1
        
        

    


if __name__ == '__main__':
    main()

