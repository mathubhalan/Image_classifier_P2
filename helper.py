# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 17:13:21 2018

@author: Mathu_Gopalan
"""

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
from PIL import Image

arch = {"vgg16":25088,
        "densenet121" :1024}
 
# Function to Load Data
def data_load (path):        
    '''
    function to load the data, transform to feed into neural network
    parameters as below
    input arugments : path, from where the data to be loaded
    output returns train, validation and test data 
    '''    
    data_dir = path
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    # Define your transforms for the training, validation, and testing sets
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    validation_transforms = transforms.Compose([transforms.Resize(256),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])

    #  Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(train_dir, transform=train_transforms)
    validation_data = datasets.ImageFolder(valid_dir, transform=validation_transforms)
    test_data = datasets.ImageFolder(test_dir ,transform = test_transforms)


    # Using the image datasets and the trainforms, define the dataloaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size =32,shuffle = True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size = 20, shuffle = True)
    
    return trainloader, validationloader, testloader

# Function to define the Neural Network Arch
    
def nn_arch(arch='vgg16',dropout=0.5, hidden_layer1 = 120,lr = 0.001,power=gpu):
    '''
    Function to define the Neural Network architecture
    Arguments : arch, dropout, hidden_layer1, lr, power
    returns the model, critieria and optimizer
    '''
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
    else:
        print("{} is not a selectable model. Kindly try with vgg16 or densenet121".format(arch))
             
        
    for param in model.parameters():
        param.requires_grad = False

        
        classifier = nn.Sequential(OrderedDict([
            ('dropout',nn.Dropout(dropout)),
            ('inputs', nn.Linear(25088, hidden_layer1)),
            ('relu1', nn.ReLU()),
            ('hidden_layer1', nn.Linear(hidden_layer1, 90)),
            ('relu2',nn.ReLU()),
            ('hidden_layer2',nn.Linear(90,80)),
            ('relu3',nn.ReLU()),
            ('hidden_layer3',nn.Linear(80,102)),
            ('output', nn.LogSoftmax(dim=1))
                          ]))
        
        
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr)
        model.cuda()
           
    return model , optimizer ,criterion 

        

# Function to Train the Neural Network
def train_network(model, criterion, optimizer, epochs = 3, print_every=20, loader=trainloader, power='gpu'):
    
    '''
    Arguments: The model, the criterion, the optimizer, epochs, dataset, and option to use a gpu or not
    Returns: Nothing
    This function trains the model over a certain number of epochs and displays the training,validation 
    and accuracy. Prints every steps
    '''
    steps = 0
    running_loss = 0

    print("--------------Training a Neural Network-------------- ")
    for e in range(epochs):
        running_loss = 0
        for ii, (inputs, labels) in enumerate(loader):
            steps += 1
            if torch.cuda.is_available() and power =='gpu':
                inputs, labels = inputs.to('cuda'), labels.to('cuda')

            optimizer.zero_grad()

            # Forward and backward passes
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if steps % print_every == 0:
                model.eval()
                validationlost = 0
                accuracy=0


                for ii, (inputs2,labels2) in enumerate(validationloader):
                    optimizer.zero_grad()
                    if torch.cuda.is_available():
                        inputs2, labels2 = inputs2.to('cuda:0') , labels2.to('cuda:0')
                        model.to('cuda:0')

                    with torch.no_grad():
                        outputs = model.forward(inputs2)
                        validationlost = criterion(outputs,labels2)
                        ps = torch.exp(outputs).data
                        equality = (labels2.data == ps.max(1)[1])
                        accuracy += equality.type_as(torch.FloatTensor()).mean()

                validationlost = validationlost / len(validationloader)
                accuracy = accuracy /len(validationloader)



                print("Epoch: {}/{}... ".format(e+1, epochs),
                      "Loss: {:.4f}".format(running_loss/print_every),
                      "Validation Lost {:.4f}".format(validationlost),
                      "Accuracy: {:.4f}".format(accuracy))
                running_loss = 0
                
    print ("---------------- Neural Network Trained -----------------")    
    print ("Number of Epochs {}".format(epochs))
    print ("Number of Steps {}".format(steps))
    

# Function to Save the Checkpoint
def save_checkpoint(path='checkpoint.pth',arch ='vgg16', hidden_layer1=120,dropout=0.5,lr=0.001,epochs=12):
    '''
    Arguments: path and hyperparameters of the network
    Returns: Nothing
    function saves the model in the defined user path    '''
    model.class_to_idx = train_data.class_to_idx
    model.cpu
    torch.save({'structure' :arch,
                'hidden_layer1':hidden_layer1,
                'dropout':dropout,
                'lr':lr,
                'nb_of_epochs':epochs,
                'state_dict':model.state_dict(),
                'class_to_idx':model.class_to_idx},
                path)
    print ("Model Saved Successfully")


# Function to Load the check point
def load_checkpoint(path='checkpoint.pth'):
    '''
    Arguments: The path of the checkpoint file
    Returns: The Neural Netowrk with all hyperparameters, weights and biases
    '''
    checkpoint = torch.load(path)
    structure = checkpoint['structure']
    hidden_layer1 = checkpoint['hidden_layer1']
    dropout = checkpoint['dropout']
    lr=checkpoint['lr']

    model,_,_ = nn_arch(structure, dropout, hidden_layer1, lr)

    model.class_to_idx = checkpoint['class_to_idx']
    model.load_state_dict(checkpoint['state_dict'])
    
    print ("Model Loaded from the checkpoint defined in {}".format(path))

# Function to Process the Image
def process_image(image):
    
    ''' 
    Argument: gets the path of Image
    Function: Scales, crops, and normalizes a PIL image for a PyTorch model,
    Returns : returns an Numpy array
    '''
    img_pil = Image.open(image)
    adjustments = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    img_tensor = adjustments(img_pil)
    
    return img_tensor 


# Function to Predict the Image
def predict(image_path, model, topk=5, power=gpu):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # Implement the code to predict the class from an image file
    if torch.cuda.is_available() and power =='gpu':
        model.to('cuda:0')
    
    img_torch = process_image(image_path)
    img_torch = img_torch.unsqueeze_(0)
    img_torch = img_torch.float()
    
    if power == 'gpu':
        with torch.no_grad():
            output = model.forward(img_torch.cuda())
    else:
        with torch.no_grad():
            output=model.forward(img_torch)
        
    probability = F.softmax(output.data,dim=1)
    
    return probability.topk(topk)



