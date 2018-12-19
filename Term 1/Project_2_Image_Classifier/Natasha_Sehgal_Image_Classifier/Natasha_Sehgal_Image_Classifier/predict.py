import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import torch, json

from torch import optim
from torch import nn

import torch.nn.functional as F
from torchvision import datasets, transforms, models
from collections import OrderedDict
import tensorflow as tf

from PIL import Image, ImageOps

def process_image(image):
    im2 = Image.open(image)
    im = im2
    new_height = 256
    new_width  = new_height * im.width // im.height
    im = im.resize((new_width, new_height), Image.ANTIALIAS)
    im = ImageOps.fit(im,(224,224),centering=(0.5, 0.5))
    np_image = np.array(im)
    np_image = (np_image - np.mean(np_image))/np.std(np_image)
    np_image = np_image.transpose((2, 0,1))
    return np_image

def predict(image_path, model, topk=5):
    criterion = nn.CrossEntropyLoss()
    np_image = process_image(image_path)
    img = torch.from_numpy(np_image).float().to(device)
    img.unsqueeze_(0) 
    model.to(device)
    model.eval()
    with torch.no_grad():
        output = model.forward(img)
    #ps = F.softmax(output,dim=1)
    ps = torch.exp(output)
    answer = ps.topk(topk) 
    probs = answer[0]
    classes = answer[1]
    return probs,classes


parser = argparse.ArgumentParser()
parser.add_argument('path_to_image',type=str,default="flowers/test/1/image_06743.jpg")
parser.add_argument('checkpoint',type=str,default="Part2_Checkpoint_1.pth")

parser.add_argument('--gpu',action="store_true")
parser.add_argument('--category_names',type=str)# default="cat_to_name.json")
parser.add_argument('--top_k',type=int,default=1) 

args = parser.parse_args()
device = torch.device('cuda:0' if args.gpu else 'cpu')

checkpoint= torch.load(args.checkpoint,map_location=lambda storage,loc:storage)  
load_model = getattr(models,checkpoint['model_name'])(pretrained=True)
#load_model= models.vgg19(pretrained=True)
load_model.class_to_idx=checkpoint['class_to_index']
load_model.classifier=checkpoint['classifier']
load_model.load_state_dict(checkpoint['state_dict'])

inv_map = {v: k for k, v in load_model.class_to_idx.items()}
topk = args.top_k
preds,classes = predict(args.path_to_image,load_model,topk)

preds = np.array(preds)
classes = np.array(classes)

x,y = [],[]

if args.category_names:
    with open(args.category_names, 'r') as f:
        cat_to_name = json.load(f)
    for i in range(topk):
        k = str(inv_map[classes[0][i]])
        x.append(cat_to_name[k])
        y.append(preds[0][i])
else:
    x,y = [inv_map[classes[0][x]] for x in range(topk)],preds

print(x,y)
