import torch
import random
def generate_vocab(filename):
    with open(filename) as vocab_filename:
        all_text = vocab_filename.read()
        unique_chars = set(all_text)
        unique_chars = sorted(list(unique_chars))
        unique_chars = ''.join(unique_chars)
        return unique_chars, len(unique_chars), all_text

def generate_bars(all_text):
    bars = all_text.split('\n[')
    return bars

def generate_bar(): return '[' + random.choice(bars)

all_letters, n_letters, all_text  = generate_vocab('../data/all_merged.txt')
bars = generate_bars(all_text)


def letter_to_index(letter):
    return all_letters.find(letter)

def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

def line_to_tensor(line):
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor

def random_bar(text):
    # from the text, pull a random bar
    print('barrrr')

if __name__ == "__main__":
    print(line_to_tensor("kanye west").shape)
    print(generate_bar())
