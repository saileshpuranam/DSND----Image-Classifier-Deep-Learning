import argparse
import data_utils
import network_utils

import matplotlib.pyplot as plt
import torch
import numpy as np
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from PIL import Image
import seaborn as sns
import json
from collections import OrderedDict


# Define All the functions required for training the model

# Function arg_parser() parses keyword arguments from the CMD
def arg_parser():
    # Define parser
    parser = argparse.ArgumentParser(description="Neural Network Settings")
    
    # Add architecture selection to parser
    parser.add_argument('--arch', 
                        type=str, 
                        help='Choose architecture from torchvision.models as str')
    
    # Add checkpoint directory to parser
    parser.add_argument('--save_dir', 
                        type=str, 
                        help='Define save directory for checkpoints as str. If not specified then model will be lost.')
    
    # Add hyperparameter tuning to parser
    parser.add_argument('--learning_rate', 
                        type=float, 
                        help='Define gradient descent learning rate as float')
    parser.add_argument('--hidden_units', 
                        type=int, 
                        help='Hidden units for DNN classifier as int')
    parser.add_argument('--epochs', 
                        type=int, 
                        help='Number of epochs for training as int')

    # Add GPU Option to parser
    parser.add_argument('--gpu', 
                        action="store_true", 
                        help='Use GPU + Cuda for calculations')
    
    # Parse args
    args = parser.parse_args()
    return args

# Function train_transformer to do training transformations on a dataset
def train_transformer(train_dir):
   # Define transformation
   train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
   # Load the Data
   train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
   return train_data

# Function test_transformer(test_dir) performs test/validation transformations on a dataset
def test_transformer(test_dir):
    # Define transformation
    test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    # Load the Data
    test_data = datasets.ImageFolder(test_dir, transform=test_transforms)
    return test_data

# Function test_transformer(test_dir) performs test/validation transformations on a dataset
def valid_transformer(valid_dir):
    # Define transformation
    valid_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])])
    # Load the Data
    valid_data = datasets.ImageFolder(valid_dir, transform=valid_transforms)
    return valid_data

# Function to create a dataloader from dataset imported
def data_loader(data, train=True):
    if train: 
        loader = torch.utils.data.DataLoader(data, batch_size=50, shuffle=True)
    else: 
        loader = torch.utils.data.DataLoader(data, batch_size=50)
    return loader

# Function check_gpu
def check_gpu(gpu_arg):
   # If gpu_arg is false then simply return the cpu device
    if not gpu_arg:
        return torch.device("cpu")
    
    # If gpu_arg then make sure to check for CUDA before assigning it
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Print result
    if device == "cpu":
        print("CUDA was not found on device, using CPU instead.")
    return device

# Loader function downloads vgg16 from Torch vision
def primaryloader_model(architecture="vgg16"):
    # Load Defaults if none specified
    if type(architecture) == type(None): 
        model = models.vgg16(pretrained=True)
        model.name = "vgg16"
        print("Network architecture specified as vgg16.")
    else: 
        exec("model = models.{}(pretrained=True)".format(architecture))
        model.name = architecture
    
    # Freeze model parameters so we can access them at a later point
    for param in model.parameters():
        param.requires_grad = False 
    return model

# Function initial_classifier(model, hidden_units) creates a classifier
def initial_classifier(model, hidden_units):
    # Check that hidden layers has been input
    if type(hidden_units) == type(None): 
        hidden_units = 4096 #hyperparamters
        print("Number of Hidden Layers specificed as 4096.")
    
    # Find Input Layers
    if model.name == "vgg16":
        input_features = model.classifier[0].in_features
    elif model.name == "resnet18":
        model_ft = models.resnet18(pretrained=True)
        number_input_features = model_ft.fc.in_features
    elif model.name == "alexnet":
        alexnet = models.alexnet(pretrained=True);
        number_input_features = alexnet.classifier[6].in_features
    
    # Define Classifier params
    classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(input_features, hidden_units, bias=True)),
                          ('relu1', nn.ReLU()),
                          ('dropout1', nn.Dropout(p=0.5)),
                          ('fc2', nn.Linear(hidden_units, 102, bias=True)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    return classifier

# Function validation(model, testloader, criterion, device) validates training against testloader to return loss and accuracy
def validation(model, testloader, criterion, device):
    test_loss = 0
    accuracy = 0
    
    for ii, (inputs, labels) in enumerate(testloader):
        
        inputs, labels = inputs.to(device), labels.to(device)
        
        output = model.forward(inputs)
        test_loss += criterion(output, labels).item()
        
        ps = torch.exp(output)
        equality = (labels.data == ps.max(dim=1)[1])
        accuracy += equality.type(torch.FloatTensor).mean()
    return test_loss, accuracy

# Function network_trainer represents the training of the network model
def network_trainer(Model, Trainloader, Testloader, Device, 
                  criterion, optimizer, Epochs, Print_every, Steps):
    # Check Model Kwarg
    if type(Epochs) == type(None):
        Epochs = 5
        print("Number of Epochs specificed as 5.")    
 
    print("Training process initializing .....\n")

    # Train Model
    for e in range(Epochs):
        running_loss = 0
        Model.train() # Technically not necessary, setting this for good measure
        
        for ii, (inputs, labels) in enumerate(Trainloader):
            Steps += 1
            
            inputs, labels = inputs.to(Device), labels.to(Device)
            
            optimizer.zero_grad()
            
            # Forward and backward passes
            outputs = Model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
            running_loss += loss.item()
        
            if Steps % Print_every == 0:
                Model.eval()

                with torch.no_grad():
                    valid_loss, accuracy = validation(Model, Testloader, criterion, Device)
            
                print("Epoch: {}/{} | ".format(e+1, Epochs),
                     "Training Loss: {:.4f} | ".format(running_loss/Print_every),
                     "Validation Loss: {:.4f} | ".format(valid_loss/len(Testloader)),
                     "Validation Accuracy: {:.4f}".format(accuracy/len(Testloader)))
            
                running_loss = 0
                Model.train()

    return Model

#Function validate_model(Model, Testloader, Device) validate the above model on test data images
def validate_model(Model, Testloader, Device):
   # Do validation on the test set
    correct = 0
    total = 0
    with torch.no_grad():
        Model.eval()
        for data in Testloader:
            images, labels = data
            images, labels = images.to(Device), labels.to(Device)
            outputs = Model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    print('Accuracy achieved by the network on test images is: %d%%' % (100 * correct / total))

# Function initial_checkpoint(Model, Save_Dir, Train_data) saves the model at a defined checkpoint
def initial_checkpoint(Model, Save_Dir, Train_data,Epochs,Learning_rate):
       
    # Save model at checkpoint
    if type(Save_Dir) == type(None):
        print("Model checkpoint directory not specified, model will not be saved.")
    else:
         Model.class_to_idx = Train_data.class_to_idx
            
         # Create checkpoint dictionary
         checkpoint = {'architecture':Model.name,
                       'classifier':Model.classifier,
                       'state_dict':Model.state_dict(),
                       'epochs':Epochs,
                       'learning_rate':Learning_rate,
                       'class_to_idx':Model.class_to_idx
                        }
            
         # Save checkpoint
         torch.save(checkpoint, 'my_checkpoint.pth')
        

# Main Function which executes all pre defined functions above to have the trained model ready
def main():
     
    # Get Keyword Args for Training
    args = arg_parser()
    
    # Set directory for training
    data_dir = 'flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Pass transforms in, then create trainloader
    train_data = train_transformer(train_dir)
    valid_data = valid_transformer(valid_dir)
    test_data = test_transformer(test_dir)
    
    trainloader = data_loader(train_data)
    validloader = data_loader(valid_data, train=False)
    testloader = data_loader(test_data, train=False)
    
    # Load Model
    model = primaryloader_model(architecture=args.arch)
    
    # Build Classifier
    model.classifier = initial_classifier(model, 
                                         hidden_units=args.hidden_units)
     
    # Check for GPU
    device = check_gpu(gpu_arg=args.gpu);
    
    # Send model to device
    model.to(device);
    
    # Check for learnrate args
    if type(args.learning_rate) == type(None):
        learning_rate = 0.001
        print("Learning rate specificed as 0.001")
    else: learning_rate = args.learning_rate
    
    # Define loss and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate)
    
    # Define deep learning method
    print_every = 30
    steps = 0
    

    
    # Train the classifier layers using backpropogation
    trained_model = network_trainer(model, trainloader, validloader, 
                                  device, criterion, optimizer, args.epochs, 
                                  print_every, steps)
    
    print("\nTraining process is now complete!!")
    
    # Validate the model
    validate_model(trained_model, testloader, device)
    
    # Save the model
    initial_checkpoint(trained_model, args.save_dir, train_data, args.epochs,args.learning_rate)


# =============================================================================
# Run Program
# =============================================================================
if __name__ == '__main__': 
    main()