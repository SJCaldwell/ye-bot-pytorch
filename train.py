import torch
from torch import nn, optim
from dataset.dataset import random_training_example, n_letters
from model.model import RNN
import numpy as np
import time
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from tqdm import tqdm

# Set Seed
np.random.seed(420)

EPOCHS = 100000
print_every = 5000
plot_every = 500

INPUT_SIZE = 100
OUTPUT_SIZE = 100
HIDDEN_SIZE = 128

crit = nn.CrossEntropyLoss()
learning_rate = 5e-4

model = RNN(n_letters, HIDDEN_SIZE, n_letters)

if torch.cuda.is_available():
    device = torch.device('cuda')
    print('Using GPU')
else:
    device = torch.device('cpu')
    print('Using CPU')
model.to(device)

def time_since(since):
    now = time.time()
    s = now - since
    m = math.floor(s/60)
    s -= m * 60
    return '%dm %ds' % (m,s)

def train(input_line_tensor, target_line_tensor):
    target_line_tensor = target_line_tensor.to(device)
    target_line_tensor.unsqueeze_(-1)
    hidden = model.init_hidden()
    hidden = hidden.to(device)
    input_line_tensor = input_line_tensor.to(device)

    model.zero_grad() # gotta call for rnn

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = model(input_line_tensor[i], hidden)
        target_line_tensor[i] = target_line_tensor[i].long()
        res = torch.max(target_line_tensor[i], 1)[1]
        ex_loss = crit(output, res[0])
        if math.isnan(ex_loss):
            print('got nan')
            exit()
        loss += ex_loss
    loss.backward()
    # i think cause we called zero_grad we gotta update the gradients ourselves?
    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)

start = time.time()
total_loss = 0
all_losses = []

for iter in tqdm(range(1, EPOCHS)):
    output, loss = train(*random_training_example())
    total_loss += loss

    if iter % print_every == 0:
        print(' %s (%d %d%%) %.4f' % (time_since(start), iter, iter/EPOCHS * 100, loss))

    if iter % plot_every == 0:
        all_losses.append(total_loss/plot_every)
        total_loss = 0

plt.figure()
plt.plot(all_losses)
