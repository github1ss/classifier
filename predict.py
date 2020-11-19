#!/usr/bin/env python3
#####
# 1 . basic usage predict.py /path-to-image checkpoint
# 2 . predict.py input checkpoint --category_names cat_to_name.json
# 3 . predict.py input checkpoint --topo_k 3
# 4 . predict.py input checkpoint --gpu
import os.path
from os import path
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

from PIL import Image

from collections import OrderedDict

import argparse
import json

from print_functions import *
from time import time, sleep
from workspace_utils import keep_awake

default_data_dir = 'flowers'
num_of_classes = 102

def get_cmd_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("image_path" ,default='flowers/test/10/image_07090.jpg',
                           help="Path of image to predict example 'flowers/test/10/image_07090.jpg'")
    parser.add_argument("checkpoint",default = 'checkpoint' ,help="checkpoint file of model to use.")
    parser.add_argument("--gpu", action="store_true",default=False,help="optional default is False")
    parser.add_argument("--top_k",default="5" ,type=int,help="Top k probability to return")
    parser.add_argument("--category_names",default = "cat_to_name.json" ,help="number of epochs to use , default is'1'")

    args = parser.parse_args()
    return args
 
def load_checkpoint(filepath):
    checkpoint= torch.load(filepath)
    model = getattr(models, checkpoint['arch'])(pretrained=True)
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    model.input_size = checkpoint['input_size']
    model.output_size = checkpoint['output_size']
    #   model.hidden_layers = checkpoint['hidden_layers']  ## ???
    print("END loading checkpoint")
    return model    
    
    
def predict(image_path, model, topk=5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.double()
    model.eval()
    image = process_image(image_path)
    tensor_image = torch.from_numpy (image)
    
    tensor_image = tensor_image.to(device) ## did not help
    
    # Added following to resolve RuntimeError: expected stride to be a single integer value
    tensor_image.unsqueeze_(0)
    print("DTYPE of tensor_image:{}".format(tensor_image.dtype))
    ## output of print => DTYPE of tensor_image:torch.float64
         
    #model.to(device)
   

 # feeed the tensor to the forward process
    log_probs = model.forward(tensor_image)  ## was tensort_image
    probs = torch.exp(log_probs)
    probs, indices= probs.topk(topk, dim = 1)
    class_to_idx = model.class_to_idx
    
    print("\n INDICISE:\n{}".format(indices))
    print("\n INDICIses ZERO:\n{}".format(indices[0]))

    classes_indexed = {class_to_idx[i]: i for i in class_to_idx}
    print("\nclasses indexed:\n{}".format(classes_indexed))
    
    label_list = indices[0].tolist()
    print("\nLabel List:\n{}".format(label_list))
    
    classes_list =  list()

    for idx in label_list:
        classes_list.append(classes_indexed[idx])
        
    print("\nClasses List:\n{}".format(classes_list))
    
    return(probs[0].tolist(), classes_list)


def main():
    
    now = datetime.now()
    print ("START TIME:{}".format(now.strftime("%Y-%m-%d %H:%M:%S")))
     
    
    in_arg = get_cmd_args()
    #check_command_line_arguments(in_arg)
    
    image_path = in_arg.image_path
    
    gpu = in_arg.gpu
    top_k = in_arg.top_k
    category_names = in_arg.category_names ## to save checkpoint path
    check_point = in_arg.checkpoint
    save_dir = "save_dir"
    time_suffix = datetime.now().timestamp
    print("Running train.py with:", 
              "\n    image_path = ", image_path,
              "\n    top_k =" , top_k ,
              "\n    category_names", top_k,
              "\n    check_point", check_point
         )
    save_dir_1 = os.path.join("/home/workspace/ImageClassifier/" , save_dir)
    check_point_file = os.path.join(save_dir_1 , check_point + '.pth')
    if not os.path.isfile(check_point_file ):
        print ("ABORT! ERROR ! checkpoint  file:{} does not exist".format(check_point_file))
        exit()
    
    
    image_file_path = os.path.join("/home/workspace/ImageClassifier/", image_path)
    if not os.path.isfile(image_file_path ):
        print ("ABORT! ERROR ! image  file:{} does not exist".format(image_file_path))
        exit()
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)
    data_dir='flowers'
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'
    
    device = torch.device("cpu")
    cuda = torch.cuda.is_available()
    if (gpu and cuda):
        device = torch.device("cuda:0")
    print("CUDA:{}".format(cuda))
    
    print("DEVICE:{}".format(device))
    model=load_checkpoint("save_dir/" + check_point + '.pth')
    exit()
  
# Call to main function to run the program
if __name__ == "__main__":
    main()