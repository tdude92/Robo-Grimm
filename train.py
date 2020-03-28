import torch
import torch.nn as nn
import numpy as np
from constants import *
from models import *

def get_batches(data, batch_size, seq_length):
    n_batches = data.shape[0] // (batch_size * seq_length)
    data = data[:n_batches*SEQ_LENGTH*BATCH_SIZE]
    data = data.view(batch_size, -1)
    for i in range(1, data.shape[1]-seq_length, seq_length):
        x = data[:, (i-1):(i+seq_length-1)]
        y = data[:, i:i+seq_length]

        # One hot encode inputs
        flattened_x = x.contiguous().view(-1)
        x = []
        for c in flattened_x:
            x.append(one_hot_vectorize(int(c.item())))
        x = torch.stack(x).view(BATCH_SIZE, SEQ_LENGTH, 128).to(DEVICE)

        yield x, y


# Load and process data into batches.
with open("train_data.txt", "r") as rf:
    train_data = torch.Tensor([ord(c) for c in rf.read()]).to(DEVICE)
with open("test_data.txt", "r") as rf:
    test_data = torch.Tensor([ord(c) for c in rf.read()]).to(DEVICE)
    N_TEST_BATCHES = len(test_data) // (BATCH_SIZE * SEQ_LENGTH)

model = ChaRNN()

# Load model if available.
try:
    model.load_state_dict(torch.load("models/ChaRNN_" + MODEL_ID + ".pth"))
    print("Loaded models/ChaRNN_" + MODEL_ID + ".pth")
except FileNotFoundError:
    print("Creating models/ChaRNN_" + MODEL_ID + ".pth")
    torch.save(model.state_dict(), "models/ChaRNN_" + MODEL_ID + ".pth")

if ON_CUDA:
    model.cuda()

optimizer = torch.optim.Adam(model.parameters(), LEARN_RATE)
criterion = nn.NLLLoss()

min_loss = np.inf
for epoch in range(START_EPOCH, END_EPOCH + 1):
    hidden = None
    step = 0
    # Training
    model.train()
    for x, y in get_batches(train_data, BATCH_SIZE, SEQ_LENGTH):
        model.zero_grad()

        out, hidden = model.forward(x, hidden)

        train_loss = criterion(out, y.contiguous().view(BATCH_SIZE*SEQ_LENGTH).long())
        train_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        optimizer.step()

        hidden[0].detach_()
        hidden[1].detach_()

        step += 1
        print("\rEpoch", epoch, "   Step", step, end = "")
    
    # Validation
    model.eval()
    validation_loss = 0
    for x, y in get_batches(test_data, BATCH_SIZE, SEQ_LENGTH):
        out, hidden = model.forward(x, hidden)
        validation_loss += criterion(out, y.contiguous().view(BATCH_SIZE*SEQ_LENGTH).long())
    validation_loss /= N_TEST_BATCHES
    
    # Information dump after every epoch.
    print("\nEpoch", epoch)
    print("----------------")
    print("Training Loss:", train_loss.item())
    print("Validation Loss:", validation_loss.item())
    if train_loss.item() < min_loss:
        print(f"!!! Loss decreased: {min_loss} --> {train_loss.item()}")
        print( "    Saving...")
        torch.save(model.state_dict(), "models/ChaRNN_" + MODEL_ID + ".pth")   
    print()
    
    # Generation
    with open("out/epoch_" + str(epoch) + ".txt", "w") as wf:
        wf.write(model.generate(1000))
