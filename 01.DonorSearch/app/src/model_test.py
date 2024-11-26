import os

os.chdir(os.path.dirname(__file__))

import torch
from torchvision.models import resnet50
from torchsummary import summary

DEVICE = "cpu"

IMG_SIZE = 224

model = resnet50(weights=None)
in_features = 2048
out_features = 4
model.fc = torch.nn.Linear(in_features, out_features)
model.load_state_dict(torch.load(
        'resnet50.pth', map_location=torch.device(DEVICE)))

summary(model, (3,IMG_SIZE, IMG_SIZE))