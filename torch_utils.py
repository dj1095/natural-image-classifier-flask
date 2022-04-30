import io

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from PIL import Image

# load model
# Hyper Parameters
import Model1
from feedforward import testing_transforms

batch_size = 8
learning_rate = 0.001
model = Model1.Model1()
PATH = 'natural_images.pth'
model.load_state_dict(torch.load(PATH))
model.eval()


# Transform image
def transform_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes))
    return testing_transforms(image).unsqueeze(0)


def get_prediction(image_tensor):
    output = model(image_tensor)
    _, predicted = torch.max(output.data, 1)
    return predicted
