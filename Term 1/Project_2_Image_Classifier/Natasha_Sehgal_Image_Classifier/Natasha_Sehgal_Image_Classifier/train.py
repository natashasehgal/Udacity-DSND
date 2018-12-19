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

def training(model,trainloader,epochs,criterion,optimizer): 
    for epoch in range(epochs):
        print("EPOCH {}...".format(epoch))
        model.train()
        running_loss, running_accuracy,step = 0,0,0
        for inputs,labels in iter(trainloader):
            if(step%20==0):
              print("STEP {}...".format(step))
            step += 1
            inputs,labels = inputs.to(device),labels.to(device)
            optimizer.zero_grad()             
            outputs = model.forward(inputs)
            _, preds = torch.max(outputs,1)
            loss = criterion(outputs,labels)             
            loss.backward()
            optimizer.step() 
            running_loss+=loss.item()*inputs.size(0)
            equal = labels.data==outputs.max(dim=1)[1]
            running_accuracy += torch.sum(equal)
        running_accuracy = running_accuracy.double()
        print('epoch:{}/{}-----'.format(epoch,epochs)) #not iter
        print('Training Loss: {:.4f}'.format((running_loss/len(trainloader.dataset))),
                      ', Training Accuracy: {:.4f}'.format((running_accuracy.double()/len(trainloader.dataset))))        
        model.eval()
        with torch.no_grad():
            valid_loss,valid_accuracy = validation(model,validloader,criterion)
            print('Validation Loss: {:.4f}'.format(valid_loss),
                          ', Validation Accuracy: {:.4f}'.format(valid_accuracy))

#define validation function
def validation(model,validloader,criterion):
    valid_loss, valid_accuracy=0,0
    model = model.to(device)    
    for inputs,labels in iter(validloader):     
        inputs,labels = inputs.to(device),labels.to(device)
        output = model.forward(inputs)
        equal = (labels.data==output.max(dim=1)[1])
        valid_loss += criterion(output,labels).item()*inputs.size(0)
        valid_accuracy += torch.sum(equal)
    step_loss = valid_loss / len(validloader.dataset)
    step_acc = valid_accuracy.double() / len(validloader.dataset)  
    return step_loss, step_acc

#directory, architecture, hyperparameters
parser = argparse.ArgumentParser()
parser.add_argument('data_directory',type=str,default="flowers")
parser.add_argument('--save_dir',type=str,default=".") #curent
parser.add_argument('--arch',type=str,default="vgg19")
parser.add_argument('--learning_rate',type=float,default=0.005)
parser.add_argument('--momentum',type=float,default=0.5)
parser.add_argument('--hidden_units',type=int,default=512)
parser.add_argument('--epochs',type=int,default=10)
parser.add_argument('--gpu',action="store_true")

args = parser.parse_args()
device = torch.device('cuda:0' if args.gpu else 'cpu')

data_dir = args.data_directory
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
valid_transforms = transforms.Compose([transforms.RandomRotation(15),
                                       transforms.RandomResizedCrop(220),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406], 
                                                            [0.229, 0.224, 0.225])])
test_transforms = transforms.Compose([transforms.Resize(256),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406], 
                                                           [0.229, 0.224, 0.225])]) #dont rotate
train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
test_dataset = datasets.ImageFolder(test_dir, transform=test_transforms)
valid_dataset = datasets.ImageFolder(valid_dir, transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_dataset, batch_size=32)
validloader = torch.utils.data.DataLoader(valid_dataset, batch_size=32)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)
n_class = len(cat_to_name)
#1.
model = getattr(models,str(args.arch))(pretrained=True)
print(model)
#2.
class_in = model.classifier[0].in_features
hu = (int)(args.hidden_units)
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(class_in,hu)),                   
                          ('relu', nn.ReLU()),
                          ('dropout', nn.Dropout(p=0.5)),
                           ('fc2', nn.Linear(hu, n_class)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
model.classifier = classifier
model = model.to(device) #torch.FloatTensor or torch.cuda.FloatTensor (data type)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr = args.learning_rate, momentum=args.momentum)
training(model,trainloader,args.epochs,criterion,optimizer)
model.eval()
total,correct,step = 0,0,0
for images,labels in iter(testloader):
    image,label = images.to(device), labels.to(device) #before we can use
    with torch.no_grad(): #no need for grad during testing
        step += 1
        output = model.forward(image)
        ps = torch.exp(output) #actual prob
        _, predicted = torch.max(output.data,1)
        total += label.size(0)
        correct += (predicted == label).sum().item() 
        print("SNo {}: Accuracy of network on test images is ... {:.4f}".format(step, correct/total))
model.class_to_idx=train_dataset.class_to_idx #class to index mapping
#size,index, classifier, state_dict
save_file = str(args.save_dir) + '/Part2_Checkpoint_1.pth'
checkpoint={'model_name':args.arch,
            'input_size':class_in, 
            'output_size':n_class,
            'class_to_index':model.class_to_idx,
            'classifier':model.classifier,
            'state_dict':model.state_dict(),
            'epochs':args.epochs,
            'criterion':criterion,
            'optimizer':optimizer
           }
torch.save(checkpoint,save_file)