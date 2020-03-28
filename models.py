import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from constants import *

class ChaRNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.dropout = nn.Dropout(DROPOUT_P)
        self.lstm = nn.LSTM(128, N_HIDDEN, N_LAYERS, dropout = DROPOUT_P, batch_first = True)
        self.fc = nn.Linear(N_HIDDEN, 128)
    
    def forward(self, x, hidden):
        r_output, hidden = self.lstm(x, hidden)
        out = self.dropout(r_output)
        out = out.contiguous().view(-1, N_HIDDEN)
        out = F.log_softmax(self.fc(out), dim = 1)

        return out, hidden

    def generate(self, n_chars, primer = ""):
        hidden = None
        for primer_char in primer:
            char, hidden = self.forward(one_hot_vectorize(ord(primer_char)).view(1, 1, 128), hidden)
        
        if primer != "":
            chars = [c for c in primer]
        else:
            chars = ["ABCDEFGHIJKLMNOPQRSTUVWXYZ"[random.randint(0, 25)]]
        
        for _ in range(n_chars):
            char, hidden = self.forward(one_hot_vectorize(ord(chars[-1])).view(1, 1, 128), hidden)
            char = F.softmax(char, dim = 1).detach().cpu()[0]
            char = char.numpy().astype("float64")
            char = char / np.sum(char)
            idx = np.argmax(np.random.multinomial(1, char, 1))
            chars.append(chr(idx))
        
        return "".join(chars)
