from comet_ml import Experiment
import torch
import torchvision
from torch import nn, optim
from dataset.dataset import random_training_example, n_letters
from model.model import RNN
import numpy as np
#experiment = Experiment(api_key="ueodw9bjrtM4LGohzeyY0zNLG",
#                        project_name="ye-bot-pytorch-rnn", workspace="sjcaldwell")

# Set Seed
np.random.seed(420)

EPOCHS = 100_000
print_every = 5000

INPUT_SIZE = 100
OUTPUT_SIZE = 100
HIDDEN_SIZE = 128

crit = nn.NLLLoss()
learning_rate = 5e-4

model = RNN(n_letters, HIDDEN_SIZE, n_letters)


def train(input_line_tensor, target_line_tensor):
    target_line_tensor.unsqueeze_(-1)
    hidden = model.init_hidden()

    model.zero_grad() # gotta call for rnn

    loss = 0

    for i in range(input_line_tensor.size(0)):
        output, hidden = model(input_line_tensor[i], hidden)
        loss += crit(output, target_line_tensor[i])
    loss.backward()
    # i think cause we called zero_grad we gotta update the gradients ourselves?
    for p in model.parameters():
        p.data.add_(-learning_rate, p.grad.data)

    return output, loss.item() / input_line_tensor.size(0)

for iter in range(1, EPOCHS):
    output, loss = train(*random_training_example())
    total_loss += loss

    if iter % print_every == 0:
        print(iter, loss)

