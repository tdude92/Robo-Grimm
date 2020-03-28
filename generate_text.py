import torch
from models import *
from constants import *

n_chars = int(input("Number of Characters to Generate: "))

model = ChaRNN()
model.load_state_dict(torch.load("models/ChaRNN_" + MODEL_ID + ".pth", map_location = DEVICE))

print(model.generate(n_chars))
