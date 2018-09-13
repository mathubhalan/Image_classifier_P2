# -*- coding: utf-8 -*-
"""
Created on Thu Sep 13 20:39:03 2018

@author: Mathu_Gopalan
"""

import helper as hp
import json
import argparse
import numpy as np

parser = argparse.ArgumentParser(description='predict.py')
parser.add_argument('input_img', default='paind-project/flowers/test/1/image_06752.jpg', nargs='*', 
                    action="store", type = str, help = "mention the image path for prediction (/test/1/image_xyz.jpg)")
parser.add_argument('checkpoint', default='/home/workspace/paind-project/checkpoint.pth', nargs='*', 
                    action="store",type = str, help = "mention path from where the checkpoint to be loaded")
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('--category_names', dest="category_names", action="store", default='cat_to_name.json')
parser.add_argument('--gpu', default="gpu", action="store", dest="gpu")

pa = parser.parse_args()
path_image = pa.input_img
number_of_outputs = pa.top_k
power = pa.gpu
input_img = pa.input_img
path = pa.checkpoint


#training_loader, testing_loader, validation_loader = hp.load_data()

#load the checkpoint
hp.load_checkpoint(path)

#load the labels
with open('cat_to_name.json', 'r') as json_file:
    cat_to_name = json.load(json_file)


probabilities = hp.predict(path_image, model, number_of_outputs, power)

labels = [cat_to_name[str(index + 1)] for index in np.array(probabilities[1][0])]
probability = np.array(probabilities[0][0])

i=0
while i < number_of_outputs:
    print("{} with a probability of {}".format(labels[i], probability[i]))
    i += 1

