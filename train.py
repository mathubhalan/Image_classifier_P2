# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 20:17:23 2018

@author: Mathu_Gopalan
"""
import argparse
import helper as hp

parser = argparse.ArgumentParser(description='Train.py')

parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/", help = "mention the folder path for data")
parser.add_argument('--gpu', dest="gpu", action="store", default="gpu", help ="gpu power is enabled")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth", 
                    help = "mention the file path to save checkpoint")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.001,
                    help = "mention the learning rate for the optimizer (default 0.001)")
parser.add_argument('--dropout', dest = "dropout", action = "store", default = 0.5,
                    help = "mention the dropout value (default 05)")
parser.add_argument('--epochs', dest="epochs", action="store", type=int, default=3,
                    help = "mention the epochs value (deafult 3)")
parser.add_argument('--arch', dest="arch", action="store", default="vgg16", type = str,
                    help = "specify the neural network structure either as vgg16 or densenet121 (default vgg16)")
parser.add_argument('--hidden_units', type=int, dest="hidden_units", action="store", default=120, 
                    help = "state the units for fist hidden layer (deafult 120)")

pa = parser.parse_args()
data_path = pa.data_dir
path = pa.save_dir
lr = pa.learning_rate
structure = pa.arch
dropout = pa.dropout
hidden_layer1 = pa.hidden_units
power = pa.gpu
epochs = pa.epochs

#load the data - invoke the data_load method from helper
trainloader, v_loader, testloader = hp.data_load(data_path)

#create the model
model, optimizer, criterion = hp.nn_arch(structure,dropout,hidden_layer1,lr,power)

#train the neural network
hp.train_network(model, optimizer, criterion, epochs, 20, trainloader, power)

#save  the train network checkpoint
hp.save_checkpoint(path,structure,hidden_layer1,dropout,lr)


print("The Model is trained")

