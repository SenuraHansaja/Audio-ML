import torch
import torchvision
import os

print(os.getppid())
dfadf

def get_model():
    return torchvision.models.resnet18(pretrained=True)

def get_pred(model):
    return model(torch.rand([1,3,224,224]))

model = get_model()

for i in range(1,10000):
    get_pred(model)
    print(i)
    
    

    