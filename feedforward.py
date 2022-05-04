import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

import matplotlib.pyplot as plt
import numpy as np
import splitfolders
import time

import Model1

#splitfolders.ratio("./data/natural_images/", output="output", seed=1337, ratio=(.8, .1, .1), group_prefix=None)


def get_accuracy(dataloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for data in dataloader:
            images, labels = data
            # calculate outputs by running images through the network
            outputs = model1(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return round(correct / total, 3)


# Define Image Transforms
training_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(150),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

validation_transforms = transforms.Compose([transforms.Resize(175),
                                            transforms.CenterCrop(150),
                                            transforms.ToTensor(),
                                            transforms.Normalize([0.485, 0.456, 0.406],
                                                                 [0.229, 0.224, 0.225])])

testing_transforms = transforms.Compose([transforms.Resize(175),
                                         transforms.CenterCrop(150),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])

# Hyper Parameters
batch_size = 8
learning_rate = 0.001

num_of_epochs = 10

traindataset = datasets.ImageFolder('./output/train', transform=training_transforms)
trainloader = torch.utils.data.DataLoader(traindataset, batch_size=batch_size, shuffle=True)

valdataset = datasets.ImageFolder('./output/val', transform=validation_transforms)
valloader = torch.utils.data.DataLoader(valdataset, batch_size=batch_size, shuffle=True)

testdataset = datasets.ImageFolder('./output/test', transform=testing_transforms)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=batch_size, shuffle=True)

classes = ('Airplane', 'Car', 'Cat', 'Dog', 'Flower', 'Fruit', 'Motorbike', 'Person')

model1 = Model1.Model1()

# Loss
criterion = nn.CrossEntropyLoss()
# optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9)
optimizer = optim.Adam(model1.parameters(), lr=learning_rate)

# Train The Model


for epoch in range(num_of_epochs):  # loop over the dataset multiple times
    epoch_start_time = time.time()
    running_loss = 0.0
    correct = 0
    total = 0
    for i, data in enumerate(trainloader, 0):
        # get the inputs data is a list of [inputs, labels]
        inputs, labels = data
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        outputs = model1(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss = running_loss + loss.item()
        # for accuracy calculation
        _, predicted = torch.max(outputs.data, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    epoch_end_time = time.time()
    avg_running_loss = round(running_loss / i + 1, 3)
    time_taken = round(epoch_end_time - epoch_start_time, 3)
    # print statistics
    print(
        f'Epoch [{epoch + 1}/{num_of_epochs}], Step [{i + 1}/{len(trainloader)}], Time taken:[{time_taken} seconds], Loss: {avg_running_loss:.4f}')
print('Finished Training')
torch.save(model1.state_dict(), 'natural_images.pth')
