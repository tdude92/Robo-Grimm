import torch

# Constants
MODEL_ID    = "0"

START_EPOCH = 1
END_EPOCH   = 200
BATCH_SIZE  = 8
SEQ_LENGTH  = 64

ON_CUDA     = torch.cuda.is_available()
DEVICE      = "cuda:0" if ON_CUDA else "cpu"

N_HIDDEN    = 256
N_LAYERS    = 2
DROPOUT_P   = 0.5
LEARN_RATE  = 0.001
CLIP        = 5

# Utility Functions
def one_hot_vectorize(char):
    vec = torch.zeros(128).to(DEVICE)
    vec[char] = 1
    return vec
